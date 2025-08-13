if __name__ == "__main__":
    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------- Phase 1: head-only warmup --------
    results_phase1 = train_1_binary(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=8,
        print_lr=True,
        early_stopping_patience=4,
        enable_gradcam=True,
        gradcam_every_n_epochs=None,
        gradcam_samples=8,
        gradcam_outdir=f"./gradcam_phase1_{run_id}",
        gradcam_alpha=0.35,
        gradcam_target_layer=model.features[-1],
        gradcam_target=None,
        scheduler=None,
    )

    # Save model + metrics for Phase 1
    torch.save(model.state_dict(), f"best_model_phase1_{run_id}.pt")
    np.savez(f"metrics_phase1_{run_id}.npz",
             loss_hist_train=results_phase1[0],
             loss_hist_val=results_phase1[1],
             acc_hist_train=results_phase1[2],
             acc_hist_val=results_phase1[3],
             precision=results_phase1[4],
             recall=results_phase1[5],
             f1=results_phase1[6],
             fpr=results_phase1[7],
             tpr=results_phase1[8],
             auc=results_phase1[9])

    # -------- Phase 2: unfreeze backbone & fine-tune --------
    for p in model.features.parameters():
        p.requires_grad = True
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.requires_grad_(True)

    head_params = list(model.classifier[1].parameters())
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "classifier.1" not in n
    ]
    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": 1e-5, "weight_decay": 1e-5},
            {"params": head_params,     "lr": 3e-4, "weight_decay": 1e-4},
        ]
    )

    results_phase2 = train_1_binary(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=20,
        print_lr=True,
        early_stopping_patience=6,
        enable_gradcam=True,
        gradcam_every_n_epochs=None,
        gradcam_samples=8,
        gradcam_outdir=f"./gradcam_phase2_{run_id}",
        gradcam_alpha=0.35,
        gradcam_target_layer=model.features[-1],
        gradcam_target=None,
        scheduler=None,
    )

    # Save model + metrics for Phase 2
    torch.save(model.state_dict(), f"best_model_phase2_{run_id}.pt")
    np.savez(f"metrics_phase2_{run_id}.npz",
             loss_hist_train=results_phase2[0],
             loss_hist_val=results_phase2[1],
             acc_hist_train=results_phase2[2],
             acc_hist_val=results_phase2[3],
             precision=results_phase2[4],
             recall=results_phase2[5],
             f1=results_phase2[6],
             fpr=results_phase2[7],
             tpr=results_phase2[8],
             auc=results_phase2[9])
