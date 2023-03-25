#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 10:44:07 2023

@author: saeed
"""
from pandas import read_csv, Series, date_range
from obspy.core import event
from obspy import UTCDateTime as utc
from utils.prepareData import prepareInventory
from pyproj import Proj
import os
from datetime import timedelta as td


class feedCatalog():
    """An obspy Catalog Contractor
    """

    def __init__(self):
        pass

    def weightMapper(self, w):
        if 0.8 < w <= 1.0:
            return 0
        elif 0.6 < w <= 0.8:
            return 1
        elif 0.4 < w <= 0.6:
            return 2
        elif 0.2 < w <= 0.4:
            return 3
        elif 0.0 < w <= 0.2:
            return 4

    def setPick(self, eventPick):
        """Fill the Obspy pick object

        Args:
            eventPick (DataFrame): a DataFrame consists pick information,

        Returns:
            obspy.pick: an obspy pick object
        """
        pick = event.Pick()
        phase_score = eventPick["phase_score"]
        phase_type = eventPick["phase_type"]
        phase_time = eventPick["phase_time"]
        station_id = eventPick["station_id"]
        pick.onset = "impulsive" if phase_score > 0.7 else "emergent"
        pick.phase_hint = phase_type.upper()
        pick.update({
            "extra": {
                "nordic_pick_weight": {
                    "value": 0,
                    "namespace": "https://github.com/AI4EPS/PhaseNet"}}})
        pick.extra.nordic_pick_weight.value = self.weightMapper(
            phase_score)
        pick.time = utc(phase_time)
        net, sta, loc, chn = station_id.split(".")
        chn = chn+"Z" if "P" in pick.phase_hint.upper() else chn+"E"
        pick.waveform_id = event.WaveformStreamID(net, sta, chn)
        pick.evaluation_mode = "automatic"
        return pick

    def setPickAmp(self, eventPick):
        """Fill the Obspy pick object

        Args:
            eventPick (DataFrame): a DataFrame consists pick information,

        Returns:
            obspy.pick: an obspy pick object
        """
        pick = event.Pick()
        pick.phase_hint = "IAML"
        pick.time = utc(eventPick["phase_time"])
        net, sta, loc, chn = eventPick["station_id"].split(".")
        chn = chn+"E"
        pick.waveform_id = event.WaveformStreamID(net, sta, chn)
        pick.evaluation_mode = "automatic"
        return pick

    def setArrival(self, eventPick, pick_id):
        """Fill Obspy arrival object

        Args:
            eventPick (DataFrame): a DataFrame consists pick information,

        Returns:
            obspy.event.arrival: an obspy arrival object
        """
        arrival = event.Arrival()
        arrival.phase = eventPick["phase_type"].upper()
        arrival.time = utc(eventPick["phase_time"])
        arrival.pick_id = pick_id
        return arrival

    def setAmplitude(self, eventPick, pick_id):
        """Fill Obspy arrival object

        Args:
            eventPick (DataFrame): a DataFrame consists pick information,

        Returns:
            obspy.event.amplitude: an obspy amplitude object
        """
        amplitude = event.magnitude.Amplitude()
        amplitude.generic_amplitude = eventPick["phase_amp"]*1e-2
        amplitude.period = 1.0
        amplitude.type = "A"
        amplitude.category = "point"
        amplitude.unit = "m"
        amplitude.pick_id = pick_id
        net, sta, loc, chn = eventPick["station_id"].split(".")
        amplitude.waveform_id = event.WaveformStreamID(net, sta, chn)
        amplitude.magnitude_hint = "ML"
        amplitude.evaluation_mode = "automatic"
        amplitude.evaluation_status = "preliminary"
        return amplitude

    def setMagnitude(self, eventInfo, origin):
        """Fill Obspy magnitude object

        Args:
            mag (float): magnitude of event
            magType (str): type of magnitude
            origin (obspy.event.origin): an obspy event origin object

        Returns:
            obspy.event.magnitude: an obspy magnitude object
        """
        magnitude = event.Magnitude()
        magnitude.mag = eventInfo["magnitude"]
        magnitude.magnitude_type = "ML"
        magnitude.origin_id = origin.resource_id
        return magnitude

    def getPicksArrivalsAmplitudes(self, eventPicks):
        """extract picks, arrivals and amplitudes from input DataFrame

        Args:
            eventPicks (DataFrame): a DataFrame contains picks, arrival and
            amplitudes.

        Returns:
            tuple: a tuple contains obspy pick, arrival and amplitude objects
        """
        picks, arrivals, amplitudes = [], [], []
        for p in range(len(eventPicks)):
            pick = self.setPick(eventPicks.iloc[p])
            arrival = self.setArrival(eventPicks.iloc[p], pick.resource_id)
            picks.append(pick)
            arrivals.append(arrival)
            if "S" in eventPicks.iloc[p]["phase_type"].upper():
                pick_amp = self.setPickAmp(eventPicks.iloc[p])
                amplitude = self.setAmplitude(
                    eventPicks.iloc[p], pick_amp.resource_id)
                picks.append(pick_amp)
                amplitudes.append(amplitude)
        return picks, arrivals, amplitudes

    def setOrigin(self, eventInfo, arrivals):
        """Fill Obspy origin object

        Args:
            eventInfo (DataFrame): a DataFrame contains event information
            arrivals (obspy.arrival): an obspy arrivals

        Returns:
            obspy.origin: an obspy origin object
        """
        origin = event.Origin()
        origin.time = utc(eventInfo["time"])
        origin.latitude = eventInfo["latitude"]
        origin.longitude = eventInfo["longitude"]
        origin.depth = eventInfo["depth(m)"]
        origin.arrivals = arrivals
        origin.evaluation_mode = "automatic"
        return origin

    def setEvent(self, eventInfo, eventPicks):
        """Fill Obspy event object

        Args:
            eventInfo (DataFrame): a DataFrame contains event information
            eventPicks (DataFrame): a DataFrame contains event picks

        Returns:
            obspy.event: an obspy event
        """
        picks, arrivals, amplitudes = self.getPicksArrivalsAmplitudes(
            eventPicks)
        Event = event.Event()
        origin = self.setOrigin(eventInfo, arrivals)
        magnitude = self.setMagnitude(eventInfo, origin)
        Event.origins.append(origin)
        Event.picks = picks
        Event.amplitudes = amplitudes
        Event.magnitudes.append(magnitude)
        Event.event_type = "earthquake"
        Event.event_type_certainty = "suspected"
        info = event.base.CreationInfo()
        info.agency_id = "PhN"
        info.author = "phasenetRunner"
        info.creation_time = utc.now()
        Event.creation_info = info
        Event.preferred_origin_id = origin.resource_id
        Event.preferred_magnitude_id = magnitude.resource_id
        return Event

    def setCatalog(self, event_df, pick_df, station_df):
        """Fill Obspy catalog object

        Args:
            eventsInfoList (list): a list contains event information
            eventArrivals (list): a list contains event information

        Returns:
            _type_: _description_
        """
        catalog = event.Catalog()
        for event_index in event_df["event_index"]:
            eventInfo = event_df.iloc[event_index]
            eventPicks = pick_df[pick_df["event_index"] == event_index]
            Event = self.setEvent(eventInfo, eventPicks)
            catalog.append(Event)
        return catalog

    def exportTo(self, config, fmt="NORDIC"):

        startTime = config["starttime"]
        endTime = config["endtime"]

        startDateRange = date_range(startTime, endTime-td(days=1), freq="1D")
        endDateRange = date_range(startTime+td(days=1), endTime, freq="1D")

        catFmtSuffix = {
            "NORDIC": "out",
            "SC3ML": "xml",
            "QUAKEML": "xml"}
        proj = Proj(f"+proj=sterea\
                    +lon_0={config['center'][0]}\
                    +lat_0={config['center'][1]}\
                    +units=km")
        for st, et in zip(startDateRange, endDateRange):
            catalog = os.path.join(
                "results",
                f"catalog_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.csv")
            pick = os.path.join(
                "results",
                f"picks_{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.csv")
            catalog_df = read_csv(catalog, sep="\t")
            catalog_df.sort_values(by=["time"], inplace=True)
            pick_df = read_csv(pick, sep="\t")
            station_df, station_dict = prepareInventory(config, proj, st, et)
            catalog_df[[
                "x(km)",
                "y(km)"]] = catalog_df.apply(lambda x: Series(
                    proj(longitude=x["longitude"],
                         latitude=x["latitude"],
                         inverse=False)),
                axis=1)
            catalog_df["z(km)"] = catalog_df["depth(m)"]*1e-3
            cat = self.setCatalog(catalog_df, pick_df, station_df)
            for fmt in config["outCatFmt"]:
                cat.write(
                    os.path.join(
                        "results",
                        f"{st.strftime('%Y%m%d')}_{et.strftime('%Y%m%d')}.{catFmtSuffix[fmt]}"),
                    format=fmt,
                    high_accuracy=False)


def exporter(config):
    engine = feedCatalog()
    engine.exportTo(config)
