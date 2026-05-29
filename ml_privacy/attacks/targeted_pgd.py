import torch
import torch.nn.functional as F


def pgd_targeted(image, model, decoy_embedding, epsilon, step_size, num_steps):
    # Maximize cosine similarity to decoy_embedding under L-inf constraint (impersonation).
    single = image.dim() == 3
    if single:
        image = image.unsqueeze(0)

    device = image.device
    image = image.detach()

    decoy = decoy_embedding.detach()
    if decoy.dim() == 1:
        decoy = decoy.unsqueeze(0).expand(image.size(0), -1)

    if hasattr(model, 'eval'):
        model.eval()

    delta = torch.empty_like(image).uniform_(-epsilon, epsilon)
    delta = ((image + delta).clamp(0.0, 1.0) - image)
    delta.requires_grad_(True)

    for _ in range(num_steps):
        adv = (image + delta).clamp(0.0, 1.0)
        emb = model(adv)
        # Negative cos-sim: descending minimizes it → cos-sim to decoy increases.
        loss = -F.cosine_similarity(emb, decoy, dim=-1).mean()
        loss.backward()

        with torch.no_grad():
            delta_new = delta - step_size * delta.grad.sign()
            delta_new = delta_new.clamp(-epsilon, epsilon)
            delta_new = (image + delta_new).clamp(0.0, 1.0) - image

        delta = delta_new.requires_grad_(True)

    adv = (image + delta).clamp(0.0, 1.0).detach()
    if single:
        adv = adv.squeeze(0)
    return adv
