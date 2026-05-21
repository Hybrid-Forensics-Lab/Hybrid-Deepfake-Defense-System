import torch


def pgd_untargeted(image, model, epsilon, step_size, num_steps):
    # Maximize L2 embedding distance between clean and perturbed image under L-inf constraint.
    single = image.dim() == 3
    if single:
        image = image.unsqueeze(0)

    device = image.device
    image = image.detach()

    if hasattr(model, 'eval'):
        model.eval()

    with torch.no_grad():
        clean_emb = model(image)

    delta = torch.empty_like(image).uniform_(-epsilon, epsilon)
    delta = ((image + delta).clamp(0.0, 1.0) - image)
    delta.requires_grad_(True)

    for _ in range(num_steps):
        adv = (image + delta).clamp(0.0, 1.0)
        emb = model(adv)
        loss = torch.linalg.vector_norm(emb - clean_emb.detach(), ord=2, dim=-1).mean()
        loss.backward()

        with torch.no_grad():
            delta_new = delta + step_size * delta.grad.sign()
            delta_new = delta_new.clamp(-epsilon, epsilon)
            delta_new = (image + delta_new).clamp(0.0, 1.0) - image

        delta = delta_new.requires_grad_(True)

    adv = (image + delta).clamp(0.0, 1.0).detach()
    if single:
        adv = adv.squeeze(0)
    return adv
