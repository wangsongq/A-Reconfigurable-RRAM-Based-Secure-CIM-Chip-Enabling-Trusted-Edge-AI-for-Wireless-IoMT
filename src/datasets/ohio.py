"""OhioT1DM XML loader used by the released training and evaluation scripts."""

import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np
import utils
import os
import glob


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML_BASE_DIR = PROJECT_ROOT / "data" / "OhioT1DM" / "2018" / "train"

def load_dataset(cfg):
    """Load one patient XML file or concatenate all training XML files."""
    if os.path.basename(cfg['xml_path']) == 'all':
        print("loading training data for all patients ...")
        xml_base_dir = cfg.get('xml_base_dir', str(DEFAULT_XML_BASE_DIR))
        xmls = os.path.join(xml_base_dir, "*.xml")
        print("Looking for XML files at:", xmls)
        xml_paths = glob.glob(xmls)
        assert len(xml_paths) > 0, f"No XML files found at {xmls}"
        tups = []
        for xml_path in xml_paths:
            cfg['xml_path'] = xml_path
            tups.append(load_data(cfg))
        x_train = np.concatenate([t[0] for t in tups], axis=0)
        y_train = np.concatenate([t[1] for t in tups], axis=0)
        x_valid = np.concatenate([t[2] for t in tups], axis=0)
        y_valid = np.concatenate([t[3] for t in tups], axis=0)
        x_test = np.concatenate([t[4] for t in tups], axis=0)
        y_test = np.concatenate([t[5] for t in tups], axis=0)

        cfg['xml_path'] = 'all'
        return x_train, y_train, x_valid, y_valid, x_test, y_test
    else:
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(cfg)
        return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_data(cfg):
    """Load a single XML file, split it, and apply the configured scale factor."""
    xml_path        = cfg['xml_path']
    nb_past_steps   = int(cfg['nb_past_steps'])
    nb_future_steps = int(cfg['nb_future_steps'])
    train_fraction  = float(cfg['train_fraction'])
    valid_fraction  = float(cfg['valid_fraction'])
    test_fraction   = float(cfg['test_fraction'])

    xs, ys = load_glucose_data(xml_path, nb_past_steps, nb_future_steps)
    ys = np.expand_dims(ys, axis=1)

    x_train, x_valid, x_test = utils.split_data(xs, train_fraction,
            valid_fraction, test_fraction)
    y_train, y_valid, y_test = utils.split_data(ys, train_fraction,
            valid_fraction, test_fraction)

    # scale data
    scale = float(cfg['scale'])
    x_train *= scale
    y_train *= scale
    x_valid *= scale
    y_valid *= scale
    x_test  *= scale
    y_test  *= scale

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_glucose_data(xml_path, nb_past_steps, nb_future_steps):
    """Extract glucose windows and prediction targets from one XML file."""
    df_glucose_level = load_ohio_series(xml_path, "glucose_level", "value")
    dt = df_glucose_level.index.to_series().diff().dropna()
    idx_breaks = np.argwhere((dt != pd.Timedelta(5, 'm')).values)

    # This release keeps the input deliberately simple: only the glucose time
    # series is used, matching the compact model discussed in the paper.
    nd_glucose_level = df_glucose_level.values
    consecutive_segments = np.split(nd_glucose_level, idx_breaks.flatten())

    consecutive_segments = [c for c in consecutive_segments if len(c) >=
            nb_past_steps+nb_future_steps]

    sups = [utils.sequence_to_supervised(c, nb_past_steps, nb_future_steps) for
            c in consecutive_segments]

    xss = [sup[0] for sup in sups]
    yss = [sup[1] for sup in sups]

    xs = np.concatenate(xss)
    ys = np.concatenate(yss)

    return np.expand_dims(xs, axis=2), ys

def load_ohio_series(xml_path, variate_name, attribute, time_attribue="ts"):
    """Parse one named time series from the OhioT1DM XML structure."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for child in root:
        if child.tag == variate_name:
            dates = []
            values = []
            for event in child:
                ts = event.attrib[time_attribue]
                date = pd.to_datetime(ts, format='%d-%m-%Y %H:%M:%S')
                date = date.replace(second=0)
                value = float(event.attrib[attribute])
                dates.append(date)
                values.append(value)
            index = pd.DatetimeIndex(dates)
            series = pd.Series(values, index=index)
            return series
