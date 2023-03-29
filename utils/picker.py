#!/usr/bin/env python3

from datetime import timedelta as td
from pandas import date_range, DataFrame, Series
from utils.prepareData import (prepareWaveforms,
                               prepareInventory,
                               readPicks,
                               applyGaMMaConfig)
import os
from tqdm import tqdm
from gamma.utils import association  # convert_picks_csv, from_seconds
from pyproj import Proj


def runPhaseNet(config):

    startTime = config["starttime"]
    endTime = config["endtime"]

    startDateRange = date_range(startTime, endTime-td(days=1), freq="1D")
    endDateRange = date_range(startTime+td(days=1), endTime, freq="1D")

    proj = Proj(f"+proj=sterea\
                +lon_0={config['center'][0]}\
                +lat_0={config['center'][1]}\
                +units=km")

    # Loop over one day data
    for st, et in zip(startDateRange, endDateRange):

        # Prepare One-day-length data
        data_exists = prepareWaveforms(st, et)
        if not data_exists:
            continue

        # Apply PhaseNet predict method
        pick_outfile = f"{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}"
        min_p_prob = config["min_p_prob"]
        min_s_prob = config["min_s_prob"]
        cmd = f"phasenet/runner.sh {pick_outfile} {min_p_prob} {min_s_prob}"
        os.system(cmd)

        # Create DataFrame for stations and picks
        station_df, station_dict = prepareInventory(config, proj, st, et)
        pick_df = readPicks(pick_outfile)

        # Apply GaMMa configuration
        config = applyGaMMaConfig(config)

        # Removes picks without amplitude if amplitude flag is set to True
        if config["use_amplitude"]:
            pick_df = pick_df[pick_df["phase_amplitude"] != -1]

        # Rum GaMMa associator
        event_index0 = 0
        assignments = []
        pbar = tqdm(1)
        catalogs, assignments = association(
            pick_df,
            station_df,
            config,
            event_index0,
            config["method"],
            pbar=pbar)
        event_index0 += len(catalogs)

        if event_index0 == 0:
            continue

        # Create catalog
        catalog_csv = os.path.join(
            "results",
            f"catalog_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.csv")
        catalogs = DataFrame(
            catalogs,
            columns=["time"]+config["dims"]+[
                "magnitude",
                "sigma_time",
                "sigma_amp",
                "cov_time_amp",
                "event_index",
                "gamma_score"])
        catalogs[[
            "longitude",
            "latitude"]] = catalogs.apply(lambda x: Series(
                proj(longitude=x["x(km)"],
                     latitude=x["y(km)"],
                     inverse=True)),
            axis=1)
        catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x*1e3)
        with open(catalog_csv, "w") as fp:
            catalogs.to_csv(
                fp,
                sep="\t",
                index=False,
                float_format="%.3f",
                date_format='%Y-%m-%dT%H:%M:%S.%f',
                columns=[
                    "time",
                    "magnitude",
                    "longitude",
                    "latitude",
                    "depth(m)",
                    "sigma_time",
                    "sigma_amp",
                    "cov_time_amp",
                    "event_index",
                    "gamma_score"])

        # Add assignment to picks
        assignments_csv = os.path.join(
            "results",
            f"assignments_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.csv")
        assignments = DataFrame(
            assignments,
            columns=[
                "pick_index",
                "event_index",
                "gamma_score"])
        with open(assignments_csv, "w") as fp:
            assignments.to_csv(
                fp,
                sep="\t",
                index=False,
                columns=[
                    "pick_index",
                    "event_index",
                    "gamma_score"])
        picks_csv = os.path.join(
            "results",
            f"picks_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.csv")
        pick_df = pick_df.join(
            assignments.set_index("pick_index")
        ).fillna(-1).astype({'event_index': int})
        with open(picks_csv, "w") as fp:
            pick_df.to_csv(
                fp,
                sep="\t",
                index=False,
                date_format='%Y-%m-%dT%H:%M:%S.%f',
                columns=[
                    "station_id",
                    "phase_time",
                    "phase_type",
                    "phase_score",
                    "phase_amp",
                    "event_index",
                    "gamma_score"])
