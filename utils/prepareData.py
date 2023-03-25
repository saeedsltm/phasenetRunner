#!/usr/bin/env python3

from obspy import read, read_inventory
import os
from pathlib import Path
from tqdm import tqdm
from pandas import DataFrame, Series, read_csv
from numpy import array


def prepareWaveforms(starttime, endtime):
    path = Path("tmp")
    path.mkdir(parents=True, exist_ok=True)

    # Read Station Data
    inv = read_inventory(os.path.join(
        "DB",
        f"{starttime.strftime('%Y%m%d')}_{endtime.strftime('%Y%m%d')}",
        "stations",
        "*.xml"))
    inv.write(os.path.join("tmp", "stations.xml"), format="STATIONXML")

    # Get Station Codes
    stations = [s.split(".")[1][:4].strip()
                for s in inv.get_contents()["stations"]]

    print("+++ Preparing raw data ...")
    with open(os.path.join("tmp", "mseed.csv"), "w") as fp:
        fp.write("fname,E,N,Z\n")
        for station in tqdm(stations):
            st = read(os.path.join(
                "DB",
                f"{starttime.strftime('%Y%m%d')}_{endtime.strftime('%Y%m%d')}",
                "waveforms",
                f"??.{station}.*.???__{starttime.strftime('%Y%m%d')}T000000Z__{endtime.strftime('%Y%m%d')}T000000Z.mseed"))
            st.write(os.path.join("tmp", f"{station}.mseed"))
            sta = st[0].stats.station
            chn = st[0].stats.channel[:-1]
            fp.write(
                f"{sta}.mseed,{chn}E,{chn}N,{chn}Z\n")


def prepareInventory(config, proj, st, et):
    stationxml = os.path.join(
        "DB",
        f"{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}",
        "stations",
        "*.xml")
    inv = read_inventory(stationxml)
    station_df = []
    for net in inv:
        for station in net:
            for channel in station:
                station_df.append({
                    "id": f"{net.code}.{station.code}..{channel.code[:-1]}",
                    "longitude": station.longitude,
                    "latitude": station.latitude,
                    "elevation(m)": station.elevation,
                    "unit": "m/s",
                    "component": "E,N,Z",
                })
                break
    station_df = DataFrame(station_df)
    cx1 = station_df["longitude"] >= config["xlim_degree"][0]
    cx2 = station_df["longitude"] < config["xlim_degree"][1]
    cy1 = station_df["latitude"] >= config["ylim_degree"][0]
    cy2 = station_df["latitude"] < config["ylim_degree"][1]
    c = (cx1) & (cx2) & (cy1) & (cy2)
    station_df = station_df[c]
    station_df.reset_index(inplace=True, drop=True)
    station_df[["x(km)", "y(km)"]] = station_df.apply(
        lambda x: Series(
            proj(longitude=x.longitude, latitude=x.latitude)),
        axis=1)
    station_df["z(km)"] = station_df["elevation(m)"].apply(lambda x: -x*1e-3)
    station_dict = {station: (x, y) for station, x, y in zip(
        station_df["id"], station_df["x(km)"], station_df["y(km)"])}
    return station_df, station_dict


def readPicks(pick_outfile):
    pick_df = read_csv(os.path.join("results", f"{pick_outfile}.csv"))
    pick_df["id"] = pick_df["station_id"]
    pick_df["timestamp"] = pick_df["phase_time"]
    pick_df["amp"] = pick_df["phase_amplitude"]
    pick_df["phase_amp"] = pick_df["phase_amplitude"]
    pick_df["prob"] = pick_df["phase_score"]
    pick_df["type"] = pick_df["phase_type"]
    return pick_df


def applyGaMMaConfig(config):
    method = {
        "BGMM": 4,
        "GMM": 1}
    config["oversample_factor"] = method[config["method"]]

    config["vel"] = {"p": 6.0, "s": 6.0 / 1.75}
    config["dims"] = ["x(km)", "y(km)", "z(km)"]
    clon = config["center"][0]
    clat = config["center"][1]
    config["x(km)"] = (array(config["xlim_degree"]) -
                       array(clon))*config["degree2km"]
    config["y(km)"] = (array(config["ylim_degree"]) -
                       array(clat))*config["degree2km"]
    config["z(km)"] = (config["zlim_degree"][0], config["zlim_degree"][1])
    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),
        (0, config["z(km)"][1] + 1),
        (None, None),
    )

    if config["useEikonal"]:
        zz = config["zz"]
        vp = config["vp"]
        vp_vs_ratio = config["vp_vs_ratio"]
        vs = [v / vp_vs_ratio for v in vp]
        h = config["h"]
        vel = {"z": zz, "p": vp, "s": vs}
        config["eikonal"] = {"vel": vel,
                             "h": h,
                             "xlim": config["x(km)"],
                             "ylim": config["y(km)"],
                             "zlim": config["z(km)"]}
    return config
