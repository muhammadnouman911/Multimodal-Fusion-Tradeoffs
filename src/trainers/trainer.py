import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from utils.losses import OrthogonalProjectionLoss

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM requires two steps")

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
    def forward(self, x, labels):
        batch_centers = self.centers[labels]
        return F.mse_loss(x, batch_centers)

class Trainer:
    def __init__(self, model, config, teacher=None):
        self.model = model.to(config.device)
        self.teacher = teacher.to(config.device) if teacher else None
        if self.teacher: self.teacher.eval()
        
        self.config = config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.opl = OrthogonalProjectionLoss()
        self.center_loss = CenterLoss(config.resolved_num_classes, config.embedding_dim).to(config.device)
        
        # SAM Optimizer for 0.97+ Robustness
        base_optimizer = optim.AdamW
        self.optimizer = SAM(list(self.model.parameters()) + list(self.center_loss.parameters()), 
                             base_optimizer, rho=0.05, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer.base_optimizer, T_0=50, T_mult=2)

    def train_epoch(self, loader, alpha=1.0, epoch=0):
        self.model.train()
        
        # Dynamic GRL Alpha Annealing (Epoch 10-30)
        current_grl_alpha = 0.0
        if epoch >= 10:
            anneal_factor = min(1.0, (epoch - 10) / 20.0)
            current_grl_alpha = self.config.grl_alpha * anneal_factor
        
        if hasattr(self.model, 'grl'): self.model.grl.alpha = current_grl_alpha
        
        total_loss = 0
        it = tqdm(loader, leave=False)
        for audio, face, labels, lang_labels in it:
            audio, face, labels = audio.to(self.config.device), face.to(self.config.device), labels.to(self.config.device)
            lang_labels = lang_labels.to(self.config.device)
            
            # Mixup Augmentation
            do_mixup = self.config.mixup_alpha > 0 and np.random.rand() < 0.5
            if do_mixup:
                lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
                index = torch.randperm(audio.size(0)).to(audio.device)
                audio = lam * audio + (1 - lam) * audio[index]
                face = lam * face + (1 - lam) * face[index]
                labels_a, labels_b = labels, labels[index]
                lang_a, lang_b = lang_labels, lang_labels[index]
            
            def calculate_loss(model_out, lbls, lng_lbls, l_a=None, idx=None, t_out=None):
                if do_mixup:
                    # 1. Standard CE (Mixup)
                    loss_ce_std = l_a * self.criterion(model_out["fusion_logits"], lbls) + (1 - l_a) * self.criterion(model_out["fusion_logits"], lbls[idx])
                    # 2. ArcFace AF (Mixup)
                    loss_ce_af  = l_a * self.criterion(model_out["fusion_af"], lbls) + (1 - l_a) * self.criterion(model_out["fusion_af"], lbls[idx])
                    # 3. OPL & Lang (Mixup)
                    loss_opl    = l_a * self.opl(model_out["fusion_embed"], lbls) + (1 - l_a) * self.opl(model_out["fusion_embed"], lbls[idx])
                    loss_lang   = l_a * self.criterion(model_out["lang_logits"], lng_lbls) + (1 - l_a) * self.criterion(model_out["lang_logits"], lng_lbls[idx])
                else:
                    loss_ce_std = self.criterion(model_out["fusion_logits"], lbls)
                    loss_ce_af  = self.criterion(model_out["fusion_af"], lbls)
                    loss_opl    = self.opl(model_out["fusion_embed"], lbls)
                    loss_lang   = self.criterion(model_out["lang_logits"], lng_lbls)

                # Identity Loss Weighting (Annealed)
                # Epoch < 10: 100% Standard CE
                # Epoch >= 10: 70% Standard / 30% ArcFace
                if epoch < 10 or self.config.arcface_s == 0:
                    loss_id = loss_ce_std
                else:
                    loss_id = 0.7 * loss_ce_std + 0.3 * loss_ce_af
                loss_proxy  = F.mse_loss(model_out["face_embed"], model_out["audio_embed"])
                loss_center = self.center_loss(model_out["fusion_embed"], lbls)
                
                total_loss = loss_id + alpha * loss_opl + current_grl_alpha * loss_lang + \
                             self.config.proxy_loss_weight * loss_proxy + self.config.center_loss_weight * loss_center
                             
                # Knowledge Distillation (Tier 2, item 6)
                # Transfer "visual knowledge" from multimodal teacher to audio student
                if t_out is not None:
                    # Target: Multimodal logits from teacher | Input: Audio-only logits from student
                    p_s = F.log_softmax(model_out["audio_logits"] / self.config.distill_temp, dim=-1)
                    p_t = F.softmax(t_out["fusion_logits"].detach() / self.config.distill_temp, dim=-1)
                    loss_distill = F.kl_div(p_s, p_t, reduction='batchmean') * (self.config.distill_temp ** 2)
                    total_loss += self.config.distill_weight * loss_distill
                    
                return total_loss

            # --- Optimization Step ---
            teacher_out = self.teacher(face, audio) if self.teacher else None
            
            # --- SAM Step 1: Ascent ---
            out = self.model(face, audio, labels)
            loss = calculate_loss(out, labels, lang_labels, lam if do_mixup else None, index if do_mixup else None, teacher_out)
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            
            # --- SAM Step 2: Descent ---
            out_sam = self.model(face, audio, labels)
            loss_sam = calculate_loss(out_sam, labels, lang_labels, lam if do_mixup else None, index if do_mixup else None, teacher_out)
            loss_sam.backward()
            self.optimizer.second_step(zero_grad=True)
            
            it.set_description(f"Ep {epoch} | L {loss.item():.3f} | GRL {current_grl_alpha:.2f}")
            total_loss += loss.item()
            
        self.scheduler.step()
        return total_loss / len(loader)
