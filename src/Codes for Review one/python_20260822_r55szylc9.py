# ============================================================
# 1. Attention comparison: CSG vs SE vs ECA vs None
# ============================================================

print("\n=== Attention Comparison ===")

for fd in [1, 3]:
    subset = f"FD{fd:03d}"

    for seed in BASE_SEEDS:
        try:
            tr, va, te = load_cmapss(fd, seed)
        except Exception as e:
            log_error(f"C-MAPSS load failed | {subset} | seed {seed} | {e}")
            continue

        for attn in ["CSG", "SE", "ECA", "None"]:
            model_name = f"NanoSentry_{attn}"

            if is_done("C-MAPSS", subset, model_name, "multi_task", seed):
                continue

            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = NanoSentry(
                    n_sensors=14,
                    d=24,
                    n_classes=3,
                    attn_type=attn
                )

                metrics = train_cmapss(
                    model,
                    tr,
                    va,
                    te,
                    multi=True,
                    num_classes=3
                )

                params, kb = count_params(model)

                save_result({
                    "dataset": "C-MAPSS",
                    "subset": subset,
                    "model": model_name,
                    "variant": "multi_task",
                    "seed": seed,
                    "fold": "",
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "score": metrics["score"],
                    "cz_rmse": metrics["cz_rmse"],
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"Attention | {model_name} | {subset} | seed {seed} | {e}")
            finally:
                cleanup()


# ============================================================
# 2. Ablation study under corrected C-MAPSS protocol
# ============================================================

print("\n=== Ablation Study ===")

ablation_variants = [
    ("full", {}),
    ("no_gate", {"no_gate": True}),
    ("no_skip", {"no_skip": True}),
    ("no_dilation", {"no_dilation": True})
]

for fd in [1, 3]:
    subset = f"FD{fd:03d}"

    for seed in BASE_SEEDS:
        try:
            tr, va, te = load_cmapss(fd, seed)
        except Exception as e:
            log_error(f"C-MAPSS ablation load failed | {subset} | seed {seed} | {e}")
            continue

        for variant_name, flags in ablation_variants:
            if is_done("C-MAPSS", subset, "NanoSentry_Ablation", variant_name, seed):
                continue

            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = NanoSentryAblation(
                    n_sensors=14,
                    d=24,
                    n_classes=3,
                    **flags
                )

                metrics = train_cmapss(
                    model,
                    tr,
                    va,
                    te,
                    multi=False,
                    num_classes=3
                )

                params, kb = count_params(model)

                save_result({
                    "dataset": "C-MAPSS",
                    "subset": subset,
                    "model": "NanoSentry_Ablation",
                    "variant": variant_name,
                    "seed": seed,
                    "fold": "",
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "score": metrics["score"],
                    "cz_rmse": metrics["cz_rmse"],
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"Ablation | {variant_name} | {subset} | seed {seed} | {e}")
            finally:
                cleanup()


# ============================================================
# 3. Improved Battery LOCO experiments
# ============================================================

print("\n=== Improved Battery LOCO ===")

try:
    battery_folds = build_battery_folds()
except Exception as e:
    battery_folds = []
    log_error(f"Battery fold construction failed | {e}")

for fold in battery_folds:
    fold_id = fold["test_cell"]

    tr = fold["tr"]
    va = fold["vl"]
    te = fold["te"]

    y_mean = fold["y_mean"]
    y_std = fold["y_std"]
    class_weight = fold["class_weight"]

    # NanoSentry multi-task
    if not is_done("Battery", fold_id, "NanoSentry", "multi_task", 42, fold_id):
        try:
            torch.manual_seed(42)
            np.random.seed(42)

            model = NanoSentry(
                n_sensors=4,
                d=24,
                n_classes=3,
                attn_type="CSG"
            )

            metrics = train_battery(
                model,
                tr,
                va,
                te,
                y_mean=y_mean,
                y_std=y_std,
                class_weight=class_weight,
                multi=True,
                num_classes=3
            )

            params, kb = count_params(model)

            save_result({
                "dataset": "Battery",
                "subset": fold_id,
                "model": "NanoSentry",
                "variant": "multi_task",
                "seed": 42,
                "fold": fold_id,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "score": metrics["score"],
                "cz_rmse": metrics["cz_rmse"],
                "acc": metrics["acc"],
                "f1": metrics["f1"],
                "params": params,
                "kb": kb
            })

        except Exception as e:
            log_error(f"Battery NanoSentry multi_task | {fold_id} | {e}")
        finally:
            cleanup()

    # NanoSentry single-task
    if not is_done("Battery", fold_id, "NanoSentry", "single_task", 42, fold_id):
        try:
            torch.manual_seed(42)
            np.random.seed(42)

            model = NanoSentry(
                n_sensors=4,
                d=24,
                n_classes=3,
                attn_type="CSG"
            )

            metrics = train_battery(
                model,
                tr,
                va,
                te,
                y_mean=y_mean,
                y_std=y_std,
                class_weight=None,
                multi=False,
                num_classes=3
            )

            params, kb = count_params(model)

            save_result({
                "dataset": "Battery",
                "subset": fold_id,
                "model": "NanoSentry",
                "variant": "single_task",
                "seed": 42,
                "fold": fold_id,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "score": metrics["score"],
                "cz_rmse": metrics["cz_rmse"],
                "acc": metrics["acc"],
                "f1": metrics["f1"],
                "params": params,
                "kb": kb
            })

        except Exception as e:
            log_error(f"Battery NanoSentry single_task | {fold_id} | {e}")
        finally:
            cleanup()

    # Ridge
    if not is_done("Battery", fold_id, "Ridge", "single_task", 42, fold_id):
        try:
            metrics = run_ridge_regression(tr, te)

            save_result({
                "dataset": "Battery",
                "subset": fold_id,
                "model": "Ridge",
                "variant": "single_task",
                "seed": 42,
                "fold": fold_id,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "score": metrics["score"],
                "cz_rmse": metrics["cz_rmse"],
                "acc": "",
                "f1": "",
                "params": 0,
                "kb": 0
            })

        except Exception as e:
            log_error(f"Battery Ridge | {fold_id} | {e}")
        finally:
            cleanup()

    # CNN-32
    if not is_done("Battery", fold_id, "CNN-32", "single_task", 42, fold_id):
        try:
            torch.manual_seed(42)
            np.random.seed(42)

            model = CNNBaseline(
                n_channels=4,
                filters=32,
                n_classes=3,
                classification=False
            )

            metrics = train_battery(
                model,
                tr,
                va,
                te,
                y_mean=y_mean,
                y_std=y_std,
                class_weight=None,
                multi=False,
                num_classes=3
            )

            params, kb = count_params(model)

            save_result({
                "dataset": "Battery",
                "subset": fold_id,
                "model": "CNN-32",
                "variant": "single_task",
                "seed": 42,
                "fold": fold_id,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "score": metrics["score"],
                "cz_rmse": metrics["cz_rmse"],
                "acc": metrics["acc"],
                "f1": metrics["f1"],
                "params": params,
                "kb": kb
            })

        except Exception as e:
            log_error(f"Battery CNN-32 | {fold_id} | {e}")
        finally:
            cleanup()

    # LSTM-64
    if not is_done("Battery", fold_id, "LSTM-64", "single_task", 42, fold_id):
        try:
            torch.manual_seed(42)
            np.random.seed(42)

            model = LSTMBaseline(
                n_channels=4,
                hidden=64,
                n_classes=3,
                classification=False
            )

            metrics = train_battery(
                model,
                tr,
                va,
                te,
                y_mean=y_mean,
                y_std=y_std,
                class_weight=None,
                multi=False,
                num_classes=3
            )

            params, kb = count_params(model)

            save_result({
                "dataset": "Battery",
                "subset": fold_id,
                "model": "LSTM-64",
                "variant": "single_task",
                "seed": 42,
                "fold": fold_id,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "score": metrics["score"],
                "cz_rmse": metrics["cz_rmse"],
                "acc": metrics["acc"],
                "f1": metrics["f1"],
                "params": params,
                "kb": kb
            })

        except Exception as e:
            log_error(f"Battery LSTM-64 | {fold_id} | {e}")
        finally:
            cleanup()


# ============================================================
# 4. Improved CWRU record-level experiments
# ============================================================

if RUN_CWRU_POLISH:
    print("\n=== Improved CWRU Record-Level Split ===")

    for seed in CWRU_SEEDS:
        try:
            tr, va, te = load_cwru_split(seed)
        except Exception as e:
            log_error(f"CWRU load failed | seed {seed} | {e}")
            continue

        models = [
            ("NanoSentry", NanoSentry(1, 24, 10, "CSG")),
            ("CNN-32", CNNBaseline(1, 32, 10, True)),
            ("LSTM-64", LSTMBaseline(1, 64, 10, True))
        ]

        for model_name, model in models:
            if is_done("CWRU", "10-class", model_name, "classifier_aug", seed):
                continue

            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                metrics = train_cwru(
                    model,
                    tr,
                    va,
                    te,
                    num_classes=10,
                    epochs=80,
                    patience=20,
                    augment=True,
                    label_smoothing=0.1
                )

                params, kb = count_params(model)

                save_result({
                    "dataset": "CWRU",
                    "subset": "10-class",
                    "model": model_name,
                    "variant": "classifier_aug",
                    "seed": seed,
                    "fold": "",
                    "rmse": "",
                    "mae": "",
                    "score": "",
                    "cz_rmse": "",
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"CWRU | {model_name} | seed {seed} | {e}")
            finally:
                cleanup()

else:
    print("\nCWRU polish skipped because GPU is not available.")


# ============================================================
# Final summary
# ============================================================

print("\n=== POLISH EXPERIMENTS FINISHED ===")
print(f"Results file: {RESULTS_FILE}")
print(f"Error file: {ERROR_FILE}")

if os.path.exists(RESULTS_FILE):
    try:
        df = pd.read_csv(RESULTS_FILE)

        print(f"Total completed polished rows: {len(df)}")

        numeric_cols = ["rmse", "mae", "score", "cz_rmse", "acc", "f1"]

        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        summary = (
            df.groupby(["dataset", "model", "variant"])
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                acc_mean=("acc", "mean"),
                acc_std=("acc", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std")
            )
            .reset_index()
        )

        print("\nSummary:")
        print(summary)

    except Exception as e:
        print(f"Could not print summary: {e}")