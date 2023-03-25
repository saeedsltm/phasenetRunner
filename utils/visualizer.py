#!/usr/bin/env python3

import os
from glob import glob
from numpy import random, unique, mean, abs, min, max, sqrt, arange, histogram
from utils.extra import handle_masked_arr
from utils.prepareData import prepareInventory
import proplot as plt
from pandas import DataFrame, read_csv, Series, date_range
from pyproj import Proj
from obspy import read, read_inventory, read_events, Stream
from obspy import UTCDateTime as utc
from datetime import timedelta as td
from tqdm import tqdm
from obspy.geodetics.base import degrees2kilometers as d2k


def plotSeismicity(config):
    startTime = config["starttime"]
    endTime = config["endtime"]

    startDateRange = date_range(startTime, endTime-td(days=1), freq="1D")
    endDateRange = date_range(startTime+td(days=1), endTime, freq="1D")

    if config["plotResults"]:
        catalogs = glob(os.path.join("results", "catalog_*.csv"))
        proj = Proj(f"+proj=sterea\
                    +lon_0={config['center'][0]}\
                    +lat_0={config['center'][1]}\
                    +units=km")
        print("+++ Plotting seismicity maps ...")
        for catalog, st, et in tqdm(zip(
                catalogs, startDateRange, endDateRange)):
            catalog = read_csv(catalog, sep="\t")
            station_df, station_dict = prepareInventory(config, proj, st, et)
            catalog[[
                "x(km)",
                "y(km)"]] = catalog.apply(lambda x: Series(
                    proj(longitude=x["longitude"],
                         latitude=x["latitude"],
                         inverse=False)),
                axis=1)
            catalog["z(km)"] = catalog["depth(m)"]*1e-3

            fig, axs = plt.subplots()
            [ax.grid(ls=":") for ax in axs]
            ax = axs[0]
            ax.format(
                ultitle=f"{len(catalog)} events from {st.strftime('%Y-%m-%d')} to {et.strftime('%Y-%m-%d')}",
                fontsize=5)
            ax.set_aspect("equal")
            cb = ax.scatter(
                catalog["x(km)"],
                catalog["y(km)"],
                c=catalog["z(km)"],
                s=catalog["magnitude"],
                cmap="viridis")
            cbar = fig.colorbar(cb)
            cbar.ax.set_ylim(cbar.ax.get_ylim()[::-1])
            cbar.set_label("Depth[km]")

            ax.plot(
                station_df["x(km)"],
                station_df["y(km)"],
                "r^", ms=10, mew=1, mec="k")
            for x, y, s in zip(
                    station_df["x(km)"],
                    station_df["y(km)"],
                    station_df["id"]):
                ax.text(x, y, s.split(".")[1])
            ax.set_xlabel("Easting [km]")
            ax.set_ylabel("Northing [km]")
            fig.save(os.path.join(
                "results",
                f"seismicity_{st.strftime('%Y%m%d')}_{st.strftime('%Y%m%d')}.png"))


def pickerTest(config):
    startTime = config["starttime"]
    endTime = config["endtime"]

    startDateRange = date_range(startTime, endTime-td(days=1), freq="1D")
    endDateRange = date_range(startTime+td(days=1), endTime, freq="1D")

    print("+++ Plotting picker test samples ...")
    for st, et in tqdm(zip(startDateRange, endDateRange)):

        stream = read(os.path.join(
            "DB",
            f"{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}",
            "waveforms",
            f"??.*..???__{st.strftime('%Y%m%d')}T000000Z__{et.strftime('%Y%m%d')}T000000Z.mseed"))
        inv = read_inventory(os.path.join(
            "DB",
            f"{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}",
            "stations",
            "*.xml"))
        stream.merge()
        stream = handle_masked_arr(stream)

        catalog = os.path.join(
            "results",
            f"catalog_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.csv")
        pick = os.path.join(
            "results",
            f"picks_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.csv")
        assignment = os.path.join(
            "results",
            f"assignments_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.csv")
        proj = Proj(f"+proj=sterea\
                    +lon_0={config['center'][0]}\
                    +lat_0={config['center'][1]}\
                    +units=km")

        catalog_df = read_csv(catalog, sep="\t")
        catalog_df.sort_values(by=["time"], inplace=True)
        pick_df = read_csv(pick, sep="\t")
        assignment_df = read_csv(assignment, sep="\t")
        station_df, station_dict = prepareInventory(config, proj, st, et)
        catalog_df[[
            "x(km)",
            "y(km)"]] = catalog_df.apply(lambda x: Series(
                proj(longitude=x["longitude"],
                     latitude=x["latitude"],
                     inverse=False)),
            axis=1)
        catalog_df["z(km)"] = catalog_df["depth(m)"]*1e-3

        for n in range(config["nTests"]):

            event_index = random.randint(len(catalog_df))
            event_picks = [pick_df.iloc[i]
                           for i in assignment_df[assignment_df["event_index"] == event_index]["pick_index"]]
            event = catalog_df.iloc[event_index]

            times = [utc(pick.phase_time) for pick in event_picks]
            first, last = min(times), max(times)

            sub = Stream()

            for station in unique(
                    [pick.station_id.split(".")[1] for pick in event_picks]):
                sub.append(stream.select(station=station, channel="??Z")[0])

            sub = sub.slice(first - 5, last + 5)

            sub = sub.copy()
            sub.detrend()
            sub.filter("bandpass", freqmin=4, freqmax=15)

            fig, axs = plt.subplots()
            ax = axs[0]
            [ax.grid(ls=":") for ax in axs]
            ax.format(
                ultitle=f"Ort={utc(event['time']).strftime('%Y-%m-%dT%H:%M:%S')}, Lon={event['longitude']:0.3f}, Lat={event['latitude']:0.3f}, Dep={event['depth(m)']*1e-3:0.3f}, Mag={event['magnitude']:0.1f}",
                fontsize=4)

            for i, trace in enumerate(sub):
                normed = trace.data - mean(trace.data)
                normed = normed / max(abs(normed))
                station_x, station_y = station_dict[trace.id[:-1]]
                y = sqrt((station_x - event["x(km)"]) ** 2 +
                         (station_y - event["y(km)"]) ** 2 +
                         event["z(km)"] ** 2)
                ax.plot(trace.times(), 5 * normed + y, lw=0.25)

            for pick in event_picks:
                station_x, station_y = station_dict[pick.station_id]
                y = sqrt((station_x - event["x(km)"]) ** 2 +
                         (station_y - event["y(km)"]) ** 2 +
                         event["z(km)"] ** 2)
                x = utc(pick.phase_time) - trace.stats.starttime
                if pick.phase_type == "P":
                    ls = '-'
                else:
                    ls = '--'
                ax.plot([x, x], [y - 10, y + 10], 'k', ls=ls, lw=0.5)

            ax.set_ylim(0)
            ax.set_xlim(0, max(trace.times()))
            ax.set_ylabel("Hypocentral distance [km]")
            ax.set_xlabel("Time [s]")

            fig.save(os.path.join(
                "results",
                f"pickerTest_{n}_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.png"))


def pickerStats(config):
    if not config["relocatedCat"].strip():
        return

    print("+++ Reading catalog ...")
    catalog = read_events(config["relocatedCat"])
    data = {}

    # Statistical metrics
    dist_P, ttim_P, resi_P, wegt_P = [], [], [], []
    dist_S, ttim_S, resi_S, wegt_S = [], [], [], []

    # Loop over catalog
    print("+++ export catalog to DataFrame ...")
    for event in tqdm(catalog):
        preferred_origin = event.preferred_origin()
        picks = event.picks
        arrivals = preferred_origin.arrivals
        for arrival in arrivals:
            pha = arrival.phase
            sta = [
                pick.waveform_id.station_code
                for pick in picks if pick.resource_id ==
                arrival.pick_id][0]
            art = [
                pick.time for pick in picks if pick.resource_id ==
                arrival.pick_id][0]
            wet = [
                int(pick.extra.get("nordic_pick_weight")["value"])
                for pick in picks if pick.resource_id ==
                arrival.pick_id][0]
            ttm = art - preferred_origin.time
            if sta not in data:
                data[sta] = {
                    "DIST_P": [],
                    "TTIM_P": [],
                    "RESI_P": [],
                    "WEGT_P": [],
                    "DIST_S": [],
                    "TTIM_S": [],
                    "RESI_S": [],
                    "WEGT_S": [],
                }
            if "P" in pha.upper():
                dist_P.append(arrival.distance)
                ttim_P.append(ttm)
                resi_P.append(arrival.time_residual)
                wegt_P.append(wet)
                data[sta]["DIST_P"].append(arrival.distance)
                data[sta]["TTIM_P"].append(ttm)
                data[sta]["RESI_P"].append(arrival.time_residual)
                data[sta]["WEGT_P"].append(wet)
            if "S" in pha.upper():
                dist_S.append(arrival.distance)
                ttim_S.append(ttm)
                resi_S.append(arrival.time_residual)
                wegt_S.append(wet)
                data[sta]["DIST_S"].append(arrival.distance)
                data[sta]["TTIM_S"].append(ttm)
                data[sta]["RESI_S"].append(arrival.time_residual)
                data[sta]["WEGT_S"].append(wet)

    df_P = DataFrame(
        {
            "DIST_P": dist_P, "TTIM_P": ttim_P,
            "RESI_P": resi_P, "WEGT_P": wegt_P
        }
    )
    df_S = DataFrame(
        {
            "DIST_S": dist_S, "TTIM_S": ttim_S,
            "RESI_S": resi_S, "WEGT_S": wegt_S
        }
    )

    df_P["DIST_P"] = d2k(df_P["DIST_P"])
    df_S["DIST_S"] = d2k(df_S["DIST_S"])

    # Travel times curve
    print("+++ Plot travel time curve ...")
    fig, axs = plt.subplots()
    [ax.grid(ls=":") for ax in axs]
    ax = axs[0]
    ax.format(
        xlabel="Distance (km)",
        ylabel="Travel time (s)",
        xlim=(0, config["maxDist"]))

    p = ax.scatter(
        df_P["DIST_P"], df_P["TTIM_P"],
        m="^", c=df_P["RESI_P"], s=5, cmap="rdylbu", vmin=-1, vmax=1,
        ec="k", ew=0.2)
    s = ax.scatter(
        df_S["DIST_S"], df_S["TTIM_S"],
        m="s", c=df_S["RESI_S"], s=5, cmap="rdylbu", vmin=-1, vmax=1,
        mec="k", mew=0.2)

    ax.colorbar(p, loc="r", label="Residuals (s)")
    # ax.colorbar(s, loc="r", label="S-residuals (s)")
    fig.save(os.path.join("results", "traveltime.png"))

    # Statistics figures
    print("+++ Plot statistics ...")
    for station in tqdm(data.keys()):

        db_P = DataFrame(
            {
                "DIST_P": data[station]["DIST_P"],
                "TTIM_P": data[station]["TTIM_P"],
                "RESI_P": data[station]["RESI_P"],
                "WEGT_P": data[station]["WEGT_P"]
            }
        )
        db_S = DataFrame(
            {
                "DIST_S": data[station]["DIST_S"],
                "TTIM_S": data[station]["TTIM_S"],
                "RESI_S": data[station]["RESI_S"],
                "WEGT_S": data[station]["WEGT_S"]
            }
        )

        avg_P = db_P["RESI_P"].mean()
        std_P = db_P["RESI_P"].std()
        avg_S = db_S["RESI_S"].mean()
        std_S = db_S["RESI_S"].std()

        db_P.dropna(inplace=True)
        db_S.dropna(inplace=True)

        fig, axs = plt.subplots(ncols=2)
        [ax.grid(ls=":") for ax in axs]
        axs.format(
            suptitle=station,
            xlabel="Time residues (s)",
            ylabel="Number of picks (#)",
        )

        W = [0, 1, 2, 3]
        C = ["gray2", "gray4", "gray6", "gray8"]
        dr = 0.05
        bins = arange(-config["minmaxRes"], config["minmaxRes"] + dr, dr)
        areas = []
        for d, l, ax in zip([db_P, db_S], ["P", "S"], axs):
            if l == "P":
                s = "$\overline{m}=$"+f"{avg_P:0.2f}" + \
                    ", $\mu=$"+f"{std_P:0.2f}"
            if l == "S":
                s = "$\overline{m}=$"+f"{avg_S:0.2f}" + \
                    ", $\mu=$"+f"{std_S:0.2f}"
            ax.format(
                ultitle=s,
                urtitle=l)
            for w, c in zip(W, C):
                df = d[d[f"WEGT_{l}"] == w]
                h, edges = histogram(df[f"RESI_{l}"], bins=bins)
                x = [mean([i, j]) for i, j in zip(edges[:-1], edges[1:])]
                area = ax.area(x, h, color=c, lw=0.5, ec="k", label=str(w))
                if l == "P":
                    areas.append(area)
        fig.legend(areas, loc="r", title="Weights", ncols=1)
        fig.save(os.path.join("results", f"{station}_stat.png"))
