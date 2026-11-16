Saved baseline models of the 30 runs (seeds 42-71), trained from scratch in every run on the same training
interactions, negative-sampling rule, validation set, and split as GIFT. Produced by scripts/04_train_baselines.sh.

rf/        Random forest (scikit-learn defaults, 100 trees, random_state=seed) on the 888 pair descriptors
           rf_seed_<s>.joblib                    fitted model (joblib.load)
           rf_seed_<s>.json                      metrics on the three benchmarks (code/train_rf_baseline.py)

gnn/       GCN, GraphSAGE, GAT link predictors (standard configuration; code/train_gnn_baselines.py)
           {gcn,sage,gat}_seed_<s>.pth           state dicts
           gnn_baselines_std_seed_<s>.json       metrics on the three benchmarks

controls/  GCN ladder controls of the contribution decomposition (code/train_ladder_controls.py)
           gcn_dot_seed_<s>.pth + control_A_seed_<s>.json      A: GCN + rotating supervision + dot-product decoder
           gcn_fusion_seed_<s>.pth + control_B_seed_<s>.json   B: GCN + rotating supervision + GIFT fusion classifier

drugban/   DrugBAN, official implementation (code/train_drugban.py, code/export_drugban_weights.py)
           drugban_seed_<s>.pth                  state dict of the validation-selected epoch
           run_<s>/dta_baselines_drugban.json    metrics on the three benchmarks
           run_<s>/drugban_dgl/lead_drugban.json lead prioritization of the driver kinases

DeepDTA and GraphDTA: ../dta_leadprio/ (models/, dta_metrics_seed_<s>.json, dta_lead_seed_<s>.json).
Main-table numbers: python code/paper_numbers.py --prop results/gift --json results/gift/paper_numbers_main_table.json
