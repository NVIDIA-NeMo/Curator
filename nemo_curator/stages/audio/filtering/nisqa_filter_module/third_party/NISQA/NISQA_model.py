# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import time
import os
from glob import glob
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from . import NISQA_lib as NL


class nisqaModel(object):
    """
    nisqaModel: Main class that loads the model and the datasets. Contains
    the training loop, prediction, and evaluation function.
    """

    def __init__(self, args):
        self.args = args

        if "mode" not in self.args:
            self.args["mode"] = "main"

        self.runinfos = {}
        self._getDevice()
        self._loadModel()
        self.args["ms_max_segments"] = None
        if not self.args.get("defer_dataloading", False):
            self._loadDatasets()
        else:
            self.ds_val = None
            self.ds_train = None
        self.args["now"] = datetime.datetime.today()

        if self.args["mode"] == "main":
            print(yaml.dump(self.args, default_flow_style=None, sort_keys=False))

    def train(self):

        if self.args["dim"] == True:
            self._train_dim()
        else:
            self._train_mos()

    def evaluate(self, mapping="first_order", do_print=True, do_plot=False):
        if self.args["dim"] == True:
            self._evaluate_dim(mapping=mapping, do_print=do_print, do_plot=do_plot)
        else:
            self._evaluate_mos(mapping=mapping, do_print=do_print, do_plot=do_plot)

    def predict_v2(self, data_dir):
        assert os.path.isdir(data_dir), "Invalid data directory passed as argument"
        assert self.args[
            "defer_dataloading"
        ], "defer_dataloading must be set to use predict_v2"
        self.args["data_dir"] = data_dir
        if self.ds_val:
            del self.ds_val
        self._loadDatasets()
        return self.predict()

    def predict_v3(self, data_dir, file_list):
        assert file_list is not None and isinstance(
            file_list, (list, tuple)
        ), "Invalid data directory passed as argument"
        assert self.args[
            "defer_dataloading"
        ], "defer_dataloading must be set to use predict_v2"
        self.args["data_dir"] = data_dir
        if self.ds_val:
            del self.ds_val
        self._loadDatasetsFolder_v2(file_list)
        return self.predict()

    def predict(self):
        
        if self.args["tr_parallel"]:
            self.model = nn.DataParallel(self.model)

        if self.args["dim"] == True:
            y_val_hat, y_val = NL.predict_dim(
                self.model,
                self.ds_val,
                self.args["tr_bs_val"],
                self.dev,
                num_workers=self.args["tr_num_workers"],
            )
        else:
            y_val_hat, y_val = NL.predict_mos(
                self.model,
                self.ds_val,
                self.args["tr_bs_val"],
                self.dev,
                num_workers=self.args["tr_num_workers"],
            )

        if not self.args.get("skip_avg", False):
            df_avg = pd.DataFrame(["Average"], columns=["deg"])
            for k, v in self.ds_val.df.items():
                if k.endswith("_pred"):
                    df_avg[k] = [self.ds_val.df[k].mean()]
                elif k == "model":
                    df_avg[k] = "NISQAv2"
            self.ds_val.df = pd.concat([self.ds_val.df, df_avg], axis=0)

        if self.args["output_dir"]:
            self.ds_val.df["model"] = self.args["name"]
            self.ds_val.df.to_csv(
                os.path.join(self.args["output_dir"], "NISQA_results.csv"), index=False
            )
        if not self.args.get("disable_print", False):
            print(self.ds_val.df.to_string(index=False))
        return self.ds_val.df

    def _train_mos(self):
        """
        Trains speech quality model.
        """
        # Initialize  -------------------------------------------------------------
        if self.args["tr_parallel"]:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.dev)

        # Runname and savepath  ---------------------------------------------------
        self.runname = self._makeRunnameAndWriteYAML()

        # Optimizer  -------------------------------------------------------------
        opt = optim.Adam(self.model.parameters(), lr=self.args["tr_lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            "min",
            verbose=True,
            threshold=0.003,
            patience=self.args["tr_lr_patience"],
        )
        earlyStp = NL.earlyStopper(self.args["tr_early_stop"])

        biasLoss = NL.biasLoss(
            self.ds_train.df.db,
            anchor_db=self.args["tr_bias_anchor_db"],
            mapping=self.args["tr_bias_mapping"],
            min_r=self.args["tr_bias_min_r"],
            do_print=(self.args["tr_verbose"] > 0),
        )

        # Dataloader    -----------------------------------------------------------
        dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args["tr_bs"],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args["tr_num_workers"],
        )

        # Start training loop   ---------------------------------------------------
        print("--> start training")
        for epoch in range(self.args["tr_epochs"]):

            tic_epoch = time.time()
            batch_cnt = 0
            loss = 0.0
            y_train = (
                self.ds_train.df[self.args["csv_mos_train"]].to_numpy().reshape(-1)
            )
            y_train_hat = np.zeros((len(self.ds_train), 1))
            self.model.train()

            # Progress bar
            if self.args["tr_verbose"] == 2:
                pbar = tqdm(
                    iterable=batch_cnt,
                    total=len(dl_train),
                    ascii=">—",
                    bar_format="{bar} {percentage:3.0f}%, {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}",
                )

            for xb_spec, yb_mos, (idx, n_wins) in dl_train:

                # Estimate batch ---------------------------------------------------
                xb_spec = xb_spec.to(self.dev)
                yb_mos = yb_mos.to(self.dev)
                n_wins = n_wins.to(self.dev)

                # Forward pass ----------------------------------------------------
                yb_mos_hat = self.model(xb_spec, n_wins)
                y_train_hat[idx] = yb_mos_hat.detach().cpu().numpy()

                # Loss ------------------------------------------------------------
                lossb = biasLoss.get_loss(yb_mos, yb_mos_hat, idx)

                # Backprop  -------------------------------------------------------
                lossb.backward()
                opt.step()
                opt.zero_grad()

                # Update total loss -----------------------------------------------
                loss += lossb.item()
                batch_cnt += 1

                if self.args["tr_verbose"] == 2:
                    pbar.set_postfix(loss=lossb.item())
                    pbar.update()

            if self.args["tr_verbose"] == 2:
                pbar.close()

            loss = loss / batch_cnt

            biasLoss.update_bias(y_train, y_train_hat)

            # Evaluate   -----------------------------------------------------------
            if self.args["tr_verbose"] > 0:
                print("\n<---- Training ---->")
            self.ds_train.df["mos_pred"] = y_train_hat
            db_results_train, r_train = NL.eval_results(
                self.ds_train.df,
                dcon=self.ds_train.df_con,
                target_mos=self.args["csv_mos_train"],
                target_ci=self.args["csv_mos_train"] + "_ci",
                pred="mos_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )

            if self.args["tr_verbose"] > 0:
                print("<---- Validation ---->")
            NL.predict_mos(
                self.model,
                self.ds_val,
                self.args["tr_bs_val"],
                self.dev,
                num_workers=self.args["tr_num_workers"],
            )
            db_results, r_val = NL.eval_results(
                self.ds_val.df,
                dcon=self.ds_val.df_con,
                target_mos=self.args["csv_mos_val"],
                target_ci=self.args["csv_mos_val"] + "_ci",
                pred="mos_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )

            r = {
                "train_r_p_mean_file": r_train["r_p_mean_file"],
                "train_rmse_map_mean_file": r_train["rmse_map_mean_file"],
                **r_val,
            }

            # Scheduler update    ---------------------------------------------
            scheduler.step(loss)
            earl_stp = earlyStp.step(r)

            # Print    --------------------------------------------------------
            ep_runtime = time.time() - tic_epoch
            print(
                "ep {} sec {:0.0f} es {} lr {:0.0e} loss {:0.4f} // "
                "r_p_tr {:0.2f} rmse_map_tr {:0.2f} // r_p {:0.2f} rmse_map {:0.2f} // "
                "best_r_p {:0.2f} best_rmse_map {:0.2f},".format(
                    epoch + 1,
                    ep_runtime,
                    earlyStp.cnt,
                    NL.get_lr(opt),
                    loss,
                    r["train_r_p_mean_file"],
                    r["train_rmse_map_mean_file"],
                    r["r_p_mean_file"],
                    r["rmse_map_mean_file"],
                    earlyStp.best_r_p,
                    earlyStp.best_rmse,
                )
            )

            # Save results and model  -----------------------------------------
            self._saveResults(
                self.model,
                self.model_args,
                opt,
                epoch,
                loss,
                ep_runtime,
                r,
                db_results,
                earlyStp.best,
            )

            # Early stopping    -----------------------------------------------
            if earl_stp:
                print(
                    "--> Early stopping. best_r_p {:0.2f} best_rmse {:0.2f}".format(
                        earlyStp.best_r_p, earlyStp.best_rmse
                    )
                )
                return

        # Training done --------------------------------------------------------
        print(
            "--> Training done. best_r_p {:0.2f} best_rmse_map {:0.2f}".format(
                earlyStp.best_r_p, earlyStp.best_rmse
            )
        )
        return

    def _train_dim(self):
        """
        Trains multidimensional speech quality model.
        """
        # Initialize  -------------------------------------------------------------
        if self.args["tr_parallel"]:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.dev)

        # Runname and savepath  ---------------------------------------------------
        self.runname = self._makeRunnameAndWriteYAML()

        # Optimizer  -------------------------------------------------------------
        opt = optim.Adam(self.model.parameters(), lr=self.args["tr_lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            "min",
            verbose=True,
            threshold=0.003,
            patience=self.args["tr_lr_patience"],
        )
        earlyStp = NL.earlyStopper_dim(self.args["tr_early_stop"])

        biasLoss_1 = NL.biasLoss(
            self.ds_train.df.db,
            anchor_db=self.args["tr_bias_anchor_db"],
            mapping=self.args["tr_bias_mapping"],
            min_r=self.args["tr_bias_min_r"],
            do_print=(self.args["tr_verbose"] > 0),
        )

        biasLoss_2 = NL.biasLoss(
            self.ds_train.df.db,
            anchor_db=self.args["tr_bias_anchor_db"],
            mapping=self.args["tr_bias_mapping"],
            min_r=self.args["tr_bias_min_r"],
            do_print=(self.args["tr_verbose"] > 0),
        )

        biasLoss_3 = NL.biasLoss(
            self.ds_train.df.db,
            anchor_db=self.args["tr_bias_anchor_db"],
            mapping=self.args["tr_bias_mapping"],
            min_r=self.args["tr_bias_min_r"],
            do_print=(self.args["tr_verbose"] > 0),
        )

        biasLoss_4 = NL.biasLoss(
            self.ds_train.df.db,
            anchor_db=self.args["tr_bias_anchor_db"],
            mapping=self.args["tr_bias_mapping"],
            min_r=self.args["tr_bias_min_r"],
            do_print=(self.args["tr_verbose"] > 0),
        )

        biasLoss_5 = NL.biasLoss(
            self.ds_train.df.db,
            anchor_db=self.args["tr_bias_anchor_db"],
            mapping=self.args["tr_bias_mapping"],
            min_r=self.args["tr_bias_min_r"],
            do_print=(self.args["tr_verbose"] > 0),
        )

        # Dataloader    -----------------------------------------------------------
        dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args["tr_bs"],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args["tr_num_workers"],
        )

        # Start training loop   ---------------------------------------------------
        print("--> start training")
        for epoch in range(self.args["tr_epochs"]):

            tic_epoch = time.time()
            batch_cnt = 0
            loss = 0.0
            y_mos = self.ds_train.df["mos"].to_numpy().reshape(-1, 1)
            y_noi = self.ds_train.df["noi"].to_numpy().reshape(-1, 1)
            y_dis = self.ds_train.df["dis"].to_numpy().reshape(-1, 1)
            y_col = self.ds_train.df["col"].to_numpy().reshape(-1, 1)
            y_loud = self.ds_train.df["loud"].to_numpy().reshape(-1, 1)
            y_train = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=1)
            y_train_hat = np.zeros((len(self.ds_train), 5))

            self.model.train()

            # Progress bar
            if self.args["tr_verbose"] == 2:
                pbar = tqdm(
                    iterable=batch_cnt,
                    total=len(dl_train),
                    ascii=">—",
                    bar_format="{bar} {percentage:3.0f}%, {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}",
                )

            for xb_spec, yb_mos, (idx, n_wins) in dl_train:

                # Estimate batch ---------------------------------------------------
                xb_spec = xb_spec.to(self.dev)
                yb_mos = yb_mos.to(self.dev)
                n_wins = n_wins.to(self.dev)

                # Forward pass ----------------------------------------------------
                yb_mos_hat = self.model(xb_spec, n_wins)
                y_train_hat[idx, :] = yb_mos_hat.detach().cpu().numpy()

                # Loss ------------------------------------------------------------
                lossb_1 = biasLoss_1.get_loss(
                    yb_mos[:, 0].view(-1, 1), yb_mos_hat[:, 0].view(-1, 1), idx
                )
                lossb_2 = biasLoss_2.get_loss(
                    yb_mos[:, 1].view(-1, 1), yb_mos_hat[:, 1].view(-1, 1), idx
                )
                lossb_3 = biasLoss_3.get_loss(
                    yb_mos[:, 2].view(-1, 1), yb_mos_hat[:, 2].view(-1, 1), idx
                )
                lossb_4 = biasLoss_4.get_loss(
                    yb_mos[:, 3].view(-1, 1), yb_mos_hat[:, 3].view(-1, 1), idx
                )
                lossb_5 = biasLoss_5.get_loss(
                    yb_mos[:, 4].view(-1, 1), yb_mos_hat[:, 4].view(-1, 1), idx
                )

                lossb = lossb_1 + lossb_2 + lossb_3 + lossb_4 + lossb_5

                # Backprop  -------------------------------------------------------
                lossb.backward()
                opt.step()
                opt.zero_grad()

                # Update total loss -----------------------------------------------
                loss += lossb.item()
                batch_cnt += 1

                if self.args["tr_verbose"] == 2:
                    pbar.set_postfix(loss=lossb.item())
                    pbar.update()

            if self.args["tr_verbose"] == 2:
                pbar.close()

            loss = loss / batch_cnt

            biasLoss_1.update_bias(
                y_train[:, 0].reshape(-1, 1), y_train_hat[:, 0].reshape(-1, 1)
            )
            biasLoss_2.update_bias(
                y_train[:, 1].reshape(-1, 1), y_train_hat[:, 1].reshape(-1, 1)
            )
            biasLoss_3.update_bias(
                y_train[:, 2].reshape(-1, 1), y_train_hat[:, 2].reshape(-1, 1)
            )
            biasLoss_4.update_bias(
                y_train[:, 3].reshape(-1, 1), y_train_hat[:, 3].reshape(-1, 1)
            )
            biasLoss_5.update_bias(
                y_train[:, 4].reshape(-1, 1), y_train_hat[:, 4].reshape(-1, 1)
            )

            # Evaluate   -----------------------------------------------------------
            self.ds_train.df["mos_pred"] = y_train_hat[:, 0].reshape(-1, 1)
            self.ds_train.df["noi_pred"] = y_train_hat[:, 1].reshape(-1, 1)
            self.ds_train.df["dis_pred"] = y_train_hat[:, 2].reshape(-1, 1)
            self.ds_train.df["col_pred"] = y_train_hat[:, 3].reshape(-1, 1)
            self.ds_train.df["loud_pred"] = y_train_hat[:, 4].reshape(-1, 1)

            if self.args["tr_verbose"] > 0:
                print("\n<---- Training ---->")
                print("--> MOS:")
            db_results_train_mos, r_train_mos = NL.eval_results(
                self.ds_train.df,
                dcon=self.ds_train.df_con,
                target_mos="mos",
                target_ci="mos_ci",
                pred="mos_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )

            if self.args["tr_verbose"] > 0:
                print("--> NOI:")
            db_results_train_noi, r_train_noi = NL.eval_results(
                self.ds_train.df,
                dcon=self.ds_train.df_con,
                target_mos="noi",
                target_ci="noi_ci",
                pred="noi_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )

            if self.args["tr_verbose"] > 0:
                print("--> DIS:")
            db_results_train_dis, r_train_dis = NL.eval_results(
                self.ds_train.df,
                dcon=self.ds_train.df_con,
                target_mos="dis",
                target_ci="dis_ci",
                pred="dis_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )

            if self.args["tr_verbose"] > 0:
                print("--> COL:")
            db_results_train_col, r_train_col = NL.eval_results(
                self.ds_train.df,
                dcon=self.ds_train.df_con,
                target_mos="col",
                target_ci="col_ci",
                pred="col_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )

            if self.args["tr_verbose"] > 0:
                print("--> LOUD:")
            db_results_train_loud, r_train_loud = NL.eval_results(
                self.ds_train.df,
                dcon=self.ds_train.df_con,
                target_mos="loud",
                target_ci="loud_ci",
                pred="loud_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )

            NL.predict_dim(
                self.model,
                self.ds_val,
                self.args["tr_bs_val"],
                self.dev,
                num_workers=self.args["tr_num_workers"],
            )

            if self.args["tr_verbose"] > 0:
                print("<---- Validation ---->")
                print("--> MOS:")
            db_results_val_mos, r_val_mos = NL.eval_results(
                self.ds_val.df,
                dcon=self.ds_val.df_con,
                target_mos="mos",
                target_ci="mos_ci",
                pred="mos_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )

            if self.args["tr_verbose"] > 0:
                print("--> NOI:")
            db_results_val_noi, r_val_noi = NL.eval_results(
                self.ds_val.df,
                dcon=self.ds_val.df_con,
                target_mos="noi",
                target_ci="noi_ci",
                pred="noi_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )
            r_val_noi = {k + "_noi": v for k, v in r_val_noi.items()}

            if self.args["tr_verbose"] > 0:
                print("--> DIS:")
            db_results_val_dis, r_val_dis = NL.eval_results(
                self.ds_val.df,
                dcon=self.ds_val.df_con,
                target_mos="dis",
                target_ci="dis_ci",
                pred="dis_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )
            r_val_dis = {k + "_dis": v for k, v in r_val_dis.items()}

            if self.args["tr_verbose"] > 0:
                print("--> COL:")
            db_results_val_col, r_val_col = NL.eval_results(
                self.ds_val.df,
                dcon=self.ds_val.df_con,
                target_mos="col",
                target_ci="col_ci",
                pred="col_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )
            r_val_col = {k + "_col": v for k, v in r_val_col.items()}

            if self.args["tr_verbose"] > 0:
                print("--> LOUD:")
            db_results_val_loud, r_val_loud = NL.eval_results(
                self.ds_val.df,
                dcon=self.ds_val.df_con,
                target_mos="loud",
                target_ci="loud_ci",
                pred="loud_pred",
                mapping="first_order",
                do_print=(self.args["tr_verbose"] > 0),
            )
            r_val_loud = {k + "_loud": v for k, v in r_val_loud.items()}

            r = {
                "train_r_p_mean_file": r_train_mos["r_p_mean_file"],
                "train_rmse_map_mean_file": r_train_mos["rmse_map_mean_file"],
                "train_r_p_mean_file_noi": r_train_noi["r_p_mean_file"],
                "train_rmse_map_mean_file_noi": r_train_noi["rmse_map_mean_file"],
                "train_r_p_mean_file_dis": r_train_dis["r_p_mean_file"],
                "train_rmse_map_mean_file_dis": r_train_dis["rmse_map_mean_file"],
                "train_r_p_mean_file_col": r_train_col["r_p_mean_file"],
                "train_rmse_map_mean_file_col": r_train_col["rmse_map_mean_file"],
                "train_r_p_mean_file_loud": r_train_loud["r_p_mean_file"],
                "train_rmse_map_mean_file_loud": r_train_loud["rmse_map_mean_file"],
                **r_val_mos,
                **r_val_noi,
                **r_val_dis,
                **r_val_col,
                **r_val_loud,
            }

            db_results = {
                "db_results_val_mos": db_results_val_mos,
                "db_results_val_noi": db_results_val_noi,
                "db_results_val_dis": db_results_val_dis,
                "db_results_val_col": db_results_val_col,
                "db_results_val_loud": db_results_val_loud,
            }

            # Scheduler update    ---------------------------------------------
            scheduler.step(loss)
            earl_stp = earlyStp.step(r)

            # Print    --------------------------------------------------------
            ep_runtime = time.time() - tic_epoch

            r_dim_mos_mean = (
                1
                / 5
                * (
                    r["r_p_mean_file"]
                    + r["r_p_mean_file_noi"]
                    + r["r_p_mean_file_col"]
                    + r["r_p_mean_file_dis"]
                    + r["r_p_mean_file_loud"]
                )
            )

            print(
                "ep {} sec {:0.0f} es {} lr {:0.0e} loss {:0.4f} // "
                "r_p_tr {:0.2f} rmse_map_tr {:0.2f} // r_dim_mos_mean {:0.2f}, r_p {:0.2f} rmse_map {:0.2f} // "
                "best_r_p {:0.2f} best_rmse_map {:0.2f},".format(
                    epoch + 1,
                    ep_runtime,
                    earlyStp.cnt,
                    NL.get_lr(opt),
                    loss,
                    r["train_r_p_mean_file"],
                    r["train_rmse_map_mean_file"],
                    r_dim_mos_mean,
                    r["r_p_mean_file"],
                    r["rmse_map_mean_file"],
                    earlyStp.best_r_p,
                    earlyStp.best_rmse,
                )
            )

            # Save results and model  -----------------------------------------
            self._saveResults(
                self.model,
                self.model_args,
                opt,
                epoch,
                loss,
                ep_runtime,
                r,
                db_results,
                earlyStp.best,
            )

            # Early stopping    -----------------------------------------------
            if earl_stp:
                print(
                    "--> Early stopping. best_r_p {:0.2f} best_rmse {:0.2f}".format(
                        earlyStp.best_r_p, earlyStp.best_rmse
                    )
                )
                return

        # Training done --------------------------------------------------------
        print(
            "--> Training done. best_r_p {:0.2f} best_rmse {:0.2f}".format(
                earlyStp.best_r_p, earlyStp.best_rmse
            )
        )
        return

    def _evaluate_mos(self, mapping="first_order", do_print=True, do_plot=False):
        """
        Evaluates the model's predictions.
        """
        print("--> MOS:")
        self.db_results, self.r = NL.eval_results(
            self.ds_val.df,
            dcon=self.ds_val.df_con,
            target_mos="mos",
            target_ci="mos_ci",
            pred="mos_pred",
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot,
        )
        if self.ds_val.df_con is None:
            print(
                "r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}".format(
                    self.r["r_p_mean_file"], self.r["rmse_mean_file"]
                )
            )
        else:
            print(
                "r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}".format(
                    self.r["r_p_mean_con"],
                    self.r["rmse_mean_con"],
                    self.r["rmse_star_map_mean_con"],
                )
            )

    def _evaluate_dim(self, mapping="first_order", do_print=True, do_plot=False):
        """
        Evaluates the predictions of a multidimensional model.
        """
        print("--> MOS:")
        self.db_results_val_mos, r_val_mos = NL.eval_results(
            self.ds_val.df,
            dcon=self.ds_val.df_con,
            target_mos="mos",
            target_ci="mos_ci",
            pred="mos_pred",
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot,
        )
        if self.ds_val.df_con is None:
            print(
                "r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}".format(
                    r_val_mos["r_p_mean_file"], r_val_mos["rmse_mean_file"]
                )
            )
        else:
            print(
                "r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}".format(
                    r_val_mos["r_p_mean_con"],
                    r_val_mos["rmse_mean_con"],
                    r_val_mos["rmse_star_map_mean_con"],
                )
            )

        print("--> NOI:")
        self.db_results_val_noi, r_val_noi = NL.eval_results(
            self.ds_val.df,
            dcon=self.ds_val.df_con,
            target_mos="noi",
            target_ci="noi_ci",
            pred="noi_pred",
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot,
        )
        if self.ds_val.df_con is None:
            print(
                "r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}".format(
                    r_val_noi["r_p_mean_file"], r_val_noi["rmse_mean_file"]
                )
            )
        else:
            print(
                "r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}".format(
                    r_val_noi["r_p_mean_con"],
                    r_val_noi["rmse_mean_con"],
                    r_val_noi["rmse_star_map_mean_con"],
                )
            )
        r_val_noi = {k + "_noi": v for k, v in r_val_noi.items()}

        print("--> DIS:")
        self.db_results_val_dis, r_val_dis = NL.eval_results(
            self.ds_val.df,
            dcon=self.ds_val.df_con,
            target_mos="dis",
            target_ci="dis_ci",
            pred="dis_pred",
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot,
        )
        if self.ds_val.df_con is None:
            print(
                "r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}".format(
                    r_val_dis["r_p_mean_file"], r_val_dis["rmse_mean_file"]
                )
            )
        else:
            print(
                "r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}".format(
                    r_val_dis["r_p_mean_con"],
                    r_val_dis["rmse_mean_con"],
                    r_val_dis["rmse_star_map_mean_con"],
                )
            )
        r_val_dis = {k + "_dis": v for k, v in r_val_dis.items()}

        print("--> COL:")
        self.db_results_val_col, r_val_col = NL.eval_results(
            self.ds_val.df,
            dcon=self.ds_val.df_con,
            target_mos="col",
            target_ci="col_ci",
            pred="col_pred",
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot,
        )
        if self.ds_val.df_con is None:
            print(
                "r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}".format(
                    r_val_col["r_p_mean_file"], r_val_col["rmse_mean_file"]
                )
            )
        else:
            print(
                "r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}".format(
                    r_val_col["r_p_mean_con"],
                    r_val_col["rmse_mean_con"],
                    r_val_col["rmse_star_map_mean_con"],
                )
            )
        r_val_col = {k + "_col": v for k, v in r_val_col.items()}

        print("--> LOUD:")
        self.db_results_val_loud, r_val_loud = NL.eval_results(
            self.ds_val.df,
            dcon=self.ds_val.df_con,
            target_mos="loud",
            target_ci="loud_ci",
            pred="loud_pred",
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot,
        )
        if self.ds_val.df_con is None:
            print(
                "r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}".format(
                    r_val_loud["r_p_mean_file"], r_val_loud["rmse_mean_file"]
                )
            )
        else:
            print(
                "r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}".format(
                    r_val_loud["r_p_mean_con"],
                    r_val_loud["rmse_mean_con"],
                    r_val_loud["rmse_star_map_mean_con"],
                )
            )
        r_val_loud = {k + "_loud": v for k, v in r_val_loud.items()}

        self.r = {
            **r_val_mos,
            **r_val_noi,
            **r_val_dis,
            **r_val_col,
            **r_val_loud,
        }

        r_mean = (
            1
            / 5
            * (
                self.r["r_p_mean_con"]
                + self.r["r_p_mean_con_noi"]
                + self.r["r_p_mean_con_col"]
                + self.r["r_p_mean_con_dis"]
                + self.r["r_p_mean_con_loud"]
            )
        )

        print("\nAverage over MOS and dimensions: r_p={:0.3f}".format(r_mean))

    def _makeRunnameAndWriteYAML(self):
        """
        Creates individual run name.
        """
        runname = self.args["name"] + "_" + self.args["now"].strftime("%y%m%d_%H%M%S%f")
        print("runname: " + runname)
        run_output_dir = os.path.join(self.args["output_dir"], runname)
        Path(run_output_dir).mkdir(parents=True, exist_ok=True)
        yaml_path = os.path.join(run_output_dir, runname + ".yaml")
        with open(yaml_path, "w") as file:
            yaml.dump(self.args, file, default_flow_style=None, sort_keys=False)

        return runname

    def _loadDatasets(self):
        if self.args["mode"] == "predict_file":
            self._loadDatasetsFile()
        elif self.args["mode"] == "predict_dir":
            self._loadDatasetsFolder()
        elif self.args["mode"] == "predict_csv":
            self._loadDatasetsCSVpredict()
        elif self.args["mode"] == "main":
            self._loadDatasetsCSV()
        else:
            raise NotImplementedError("mode not available")

    def _loadDatasetsFolder(self):
        files = glob(os.path.join(self.args["data_dir"], "*.wav"))
        files = [os.path.basename(files) for files in files]
        files.sort()
        df_val = pd.DataFrame(files, columns=["deg"])

        print("# files: {}".format(len(df_val)))
        if len(df_val) == 0:
            raise ValueError("No wav files found in data_dir")

        # creating Datasets ---------------------------------------------------
        self.ds_val = NL.SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir=self.args["data_dir"],
            filename_column="deg",
            mos_column="predict_only",
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            to_memory=None,
            to_memory_workers=None,
            seg_hop_length=self.args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
            double_ended=self.args["double_ended"],
            dim=self.args["dim"],
            filename_column_ref=None,
        )

    def _loadDatasetsFolder_v2(self, filelist):
        df_val = pd.DataFrame(filelist, columns=["deg"])
        print("# files: {}".format(len(df_val)))
        if len(df_val) == 0:
            raise ValueError("No wav files found in data_dir")

        # creating Datasets ---------------------------------------------------
        self.ds_val = NL.SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir=self.args["data_dir"],
            filename_column="deg",
            mos_column="predict_only",
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            to_memory=None,
            to_memory_workers=None,
            seg_hop_length=self.args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
            double_ended=self.args["double_ended"],
            dim=self.args["dim"],
            filename_column_ref=None,
        )

    def _loadDatasetsFile(self):
        data_dir = os.path.dirname(self.args["deg"])
        file_name = os.path.basename(self.args["deg"])
        df_val = pd.DataFrame([file_name], columns=["deg"])

        # creating Datasets ---------------------------------------------------
        self.ds_val = NL.SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir=data_dir,
            filename_column="deg",
            mos_column="predict_only",
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            to_memory=None,
            to_memory_workers=None,
            seg_hop_length=self.args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
            double_ended=self.args["double_ended"],
            dim=self.args["dim"],
            filename_column_ref=None,
        )

    def _loadDatasetsCSVpredict(self):
        """
        Loads validation dataset for prediction only.
        """
        csv_file_path = os.path.join(self.args["data_dir"], self.args["csv_file"])
        dfile = pd.read_csv(csv_file_path)
        if "csv_con" in self.args:
            csv_con_file_path = os.path.join(
                self.args["data_dir"], self.args["csv_con"]
            )
            dcon = pd.read_csv(csv_con_file_path)
        else:
            dcon = None

        # creating Datasets ---------------------------------------------------
        self.ds_val = NL.SpeechQualityDataset(
            dfile,
            df_con=dcon,
            data_dir=self.args["data_dir"],
            filename_column=self.args["csv_deg"],
            mos_column="predict_only",
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            to_memory=False,
            to_memory_workers=None,
            seg_hop_length=self.args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
            double_ended=self.args["double_ended"],
            dim=self.args["dim"],
            filename_column_ref=self.args["csv_ref"],
        )

    def _loadDatasetsCSV(self):
        """
        Loads training and validation dataset for training.
        """
        csv_file_path = os.path.join(self.args["data_dir"], self.args["csv_file"])
        dfile = pd.read_csv(csv_file_path)

        if not set(self.args["csv_db_train"] + self.args["csv_db_val"]).issubset(
            dfile.db.unique().tolist()
        ):
            missing_datasets = set(
                self.args["csv_db_train"] + self.args["csv_db_val"]
            ).difference(dfile.db.unique().tolist())
            raise ValueError("Not all dbs found in csv:", missing_datasets)

        df_train = dfile[dfile.db.isin(self.args["csv_db_train"])].reset_index()
        df_val = dfile[dfile.db.isin(self.args["csv_db_val"])].reset_index()

        if self.args["csv_con"] is not None:
            csv_con_path = os.path.join(self.args["data_dir"], self.args["csv_con"])
            dcon = pd.read_csv(csv_con_path)
            dcon_train = dcon[dcon.db.isin(self.args["csv_db_train"])].reset_index()
            dcon_val = dcon[dcon.db.isin(self.args["csv_db_val"])].reset_index()
        else:
            dcon = None
            dcon_train = None
            dcon_val = None

        print(
            "Training size: {}, Validation size: {}".format(len(df_train), len(df_val))
        )

        # creating Datasets ---------------------------------------------------
        self.ds_train = NL.SpeechQualityDataset(
            df_train,
            df_con=dcon_train,
            data_dir=self.args["data_dir"],
            filename_column=self.args["csv_deg"],
            mos_column=self.args["csv_mos_train"],
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            to_memory=self.args["tr_ds_to_memory"],
            to_memory_workers=self.args["tr_ds_to_memory_workers"],
            seg_hop_length=self.args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
            double_ended=self.args["double_ended"],
            dim=self.args["dim"],
            filename_column_ref=self.args["csv_ref"],
        )

        self.ds_val = NL.SpeechQualityDataset(
            df_val,
            df_con=dcon_val,
            data_dir=self.args["data_dir"],
            filename_column=self.args["csv_deg"],
            mos_column=self.args["csv_mos_val"],
            seg_length=self.args["ms_seg_length"],
            max_length=self.args["ms_max_segments"],
            to_memory=self.args["tr_ds_to_memory"],
            to_memory_workers=self.args["tr_ds_to_memory_workers"],
            seg_hop_length=self.args["ms_seg_hop_length"],
            transform=None,
            ms_n_fft=self.args["ms_n_fft"],
            ms_hop_length=self.args["ms_hop_length"],
            ms_win_length=self.args["ms_win_length"],
            ms_n_mels=self.args["ms_n_mels"],
            ms_sr=self.args["ms_sr"],
            ms_fmax=self.args["ms_fmax"],
            ms_channel=self.args["ms_channel"],
            double_ended=self.args["double_ended"],
            dim=self.args["dim"],
            filename_column_ref=self.args["csv_ref"],
        )

        self.runinfos["ds_train_len"] = len(self.ds_train)
        self.runinfos["ds_val_len"] = len(self.ds_val)

    def _loadModel(self):
        """
        Loads the Pytorch models with given input arguments.
        """
        # if True overwrite input arguments from pretrained model
        if self.args["pretrained_model"]:
            if os.path.isabs(self.args["pretrained_model"]):
                model_path = os.path.join(self.args["pretrained_model"])
            else:
                model_path = os.path.join(os.getcwd(), self.args["pretrained_model"])
            checkpoint = torch.load(model_path, map_location=self.dev)

            # update checkpoint arguments with new arguments
            checkpoint["args"].update(self.args)
            self.args = checkpoint["args"]

        if self.args["model"] == "NISQA_DIM":
            self.args["dim"] = True
            self.args["csv_mos_train"] = None  # column names hardcoded for dim models
            self.args["csv_mos_val"] = None
        else:
            self.args["dim"] = False

        if self.args["model"] == "NISQA_DE":
            self.args["double_ended"] = True
        else:
            self.args["double_ended"] = False
            self.args["csv_ref"] = None

        # Load Model
        self.model_args = {
            "ms_seg_length": self.args["ms_seg_length"],
            "ms_n_mels": self.args["ms_n_mels"],
            "cnn_model": self.args["cnn_model"],
            "cnn_c_out_1": self.args["cnn_c_out_1"],
            "cnn_c_out_2": self.args["cnn_c_out_2"],
            "cnn_c_out_3": self.args["cnn_c_out_3"],
            "cnn_kernel_size": self.args["cnn_kernel_size"],
            "cnn_dropout": self.args["cnn_dropout"],
            "cnn_pool_1": self.args["cnn_pool_1"],
            "cnn_pool_2": self.args["cnn_pool_2"],
            "cnn_pool_3": self.args["cnn_pool_3"],
            "cnn_fc_out_h": self.args["cnn_fc_out_h"],
            "td": self.args["td"],
            "td_sa_d_model": self.args["td_sa_d_model"],
            "td_sa_nhead": self.args["td_sa_nhead"],
            "td_sa_pos_enc": self.args["td_sa_pos_enc"],
            "td_sa_num_layers": self.args["td_sa_num_layers"],
            "td_sa_h": self.args["td_sa_h"],
            "td_sa_dropout": self.args["td_sa_dropout"],
            "td_lstm_h": self.args["td_lstm_h"],
            "td_lstm_num_layers": self.args["td_lstm_num_layers"],
            "td_lstm_dropout": self.args["td_lstm_dropout"],
            "td_lstm_bidirectional": self.args["td_lstm_bidirectional"],
            "td_2": self.args["td_2"],
            "td_2_sa_d_model": self.args["td_2_sa_d_model"],
            "td_2_sa_nhead": self.args["td_2_sa_nhead"],
            "td_2_sa_pos_enc": self.args["td_2_sa_pos_enc"],
            "td_2_sa_num_layers": self.args["td_2_sa_num_layers"],
            "td_2_sa_h": self.args["td_2_sa_h"],
            "td_2_sa_dropout": self.args["td_2_sa_dropout"],
            "td_2_lstm_h": self.args["td_2_lstm_h"],
            "td_2_lstm_num_layers": self.args["td_2_lstm_num_layers"],
            "td_2_lstm_dropout": self.args["td_2_lstm_dropout"],
            "td_2_lstm_bidirectional": self.args["td_2_lstm_bidirectional"],
            "pool": self.args["pool"],
            "pool_att_h": self.args["pool_att_h"],
            "pool_att_dropout": self.args["pool_att_dropout"],
        }

        if self.args["double_ended"]:
            self.model_args.update(
                {
                    "de_align": self.args["de_align"],
                    "de_align_apply": self.args["de_align_apply"],
                    "de_fuse_dim": self.args["de_fuse_dim"],
                    "de_fuse": self.args["de_fuse"],
                }
            )

        print("Model architecture: " + self.args["model"])
        if self.args["model"] == "NISQA":
            self.model = NL.NISQA(**self.model_args)
        elif self.args["model"] == "NISQA_DIM":
            self.model = NL.NISQA_DIM(**self.model_args)
        elif self.args["model"] == "NISQA_DE":
            self.model = NL.NISQA_DE(**self.model_args)
        else:
            raise NotImplementedError("Model not available")

        # Load weights if pretrained model is used ------------------------------------
        if self.args["pretrained_model"]:
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )
            print("Loaded pretrained model from " + self.args["pretrained_model"])
            if missing_keys:
                print("missing_keys:")
                print(missing_keys)
            if unexpected_keys:
                print("unexpected_keys:")
                print(unexpected_keys)

    def _getDevice(self):
        """
        Train on GPU if available.
        """
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")

        if "tr_device" in self.args:
            if self.args["tr_device"] == "cpu":
                self.dev = torch.device("cpu")
            elif self.args["tr_device"] == "cuda":
                self.dev = torch.device("cuda")
        print("Device: {}".format(self.dev))

        if "tr_parallel" in self.args:
            if (self.dev == torch.device("cpu")) and self.args["tr_parallel"] == True:
                self.args["tr_parallel"] == False
                print("Using CPU -> tr_parallel set to False")

    def _saveResults(
        self, model, model_args, opt, epoch, loss, ep_runtime, r, db_results, best
    ):
        """
        Save model/results in dictionary and write results csv.
        """
        if self.args["tr_checkpoint"] == "best_only":
            filename = self.runname + ".tar"
        else:
            filename = self.runname + "__" + ("ep_{:03d}".format(epoch + 1)) + ".tar"
        run_output_dir = os.path.join(self.args["output_dir"], self.runname)
        model_path = os.path.join(run_output_dir, filename)
        results_path = os.path.join(run_output_dir, self.runname + "__results.csv")
        Path(run_output_dir).mkdir(parents=True, exist_ok=True)

        results = {
            "runname": self.runname,
            "epoch": "{:05d}".format(epoch + 1),
            "filename": filename,
            "loss": loss,
            "ep_runtime": "{:0.2f}".format(ep_runtime),
            **self.runinfos,
            **r,
            **self.args,
        }

        for key in results:
            results[key] = str(results[key])

        if epoch == 0:
            self.results_hist = pd.DataFrame(results, index=[0])
        else:
            self.results_hist.loc[epoch] = results
        self.results_hist.to_csv(results_path, index=False)

        if (self.args["tr_checkpoint"] == "every_epoch") or (
            self.args["tr_checkpoint"] == "best_only" and best
        ):

            if hasattr(model, "module"):
                state_dict = model.module.state_dict()
                model_name = model.module.name
            else:
                state_dict = model.state_dict()
                model_name = model.name

            torch_dict = {
                "runname": self.runname,
                "epoch": epoch + 1,
                "model_args": model_args,
                "args": self.args,
                "model_state_dict": state_dict,
                "optimizer_state_dict": opt.state_dict(),
                "db_results": db_results,
                "results": results,
                "model_name": model_name,
            }

            torch.save(torch_dict, model_path)

        elif (self.args["tr_checkpoint"] != "every_epoch") and (
            self.args["tr_checkpoint"] != "best_only"
        ):
            raise ValueError("selected tr_checkpoint option not available")
