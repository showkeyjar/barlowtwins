import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils import data

sel_col = ['lat', 'lon', 'hour', 'p3020', 'z', 'deg0l', 'cbh', 'd2m', 'skt',
       't2m', 'sp', 'msl', 'v10', 'sf', 'u10', 'lsp', 'tp', 'cp', 'tcc', 'hcc',
       'lcc', 'sd', 'mcc', 'cape', 'wind_speed', 'wind_direction', 'alti',
       'alti0', 'alti3', 'alti1', 'alti2', 'solar_azimuth_angle', 'rad_vector',
       'earth_distance', 'norm_irradiance', 'moon_Az', 'moon_El', 'moon_Dist',
       'tp1', 'precip']

class ECDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('data'))
    
    def _add_data_infos(self, file_path, load_data):
        global sel_col
        df = pd.read_hdf(file_path, "df")
        df = df[sel_col]
        df.dropna(subset=['precip'], inplace=True)
        df['precip'] = df['precip'].astype(int)
        ds = df.drop('precip', axis=1)
        ds = ds.apply(pd.to_numeric, errors='coerce')
        ds.fillna(0, inplace=True)
        idx = -1
        # type is derived from the name of the dataset; we expect the dataset
        # name to have a name such as 'data' or 'label' to identify its type
        # we also store the shape of the data in case we need it
        if load_data:
            idx = self._add_to_cache(ds.values, file_path)
        self.data_info.append({'file_path': file_path, 'type': 'data', 'shape': ds.values.shape, 'cache_idx': idx})
        label = df['precip'].values
        if load_data:
            idx = self._add_to_cache(label, file_path)
        self.data_info.append({'file_path': file_path, 'type': 'data', 'label': label.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        global sel_col
        df = pd.read_hdf(file_path, "df")
        df = df[sel_col]
        df.dropna(subset=['precip'], inplace=True)
        df['precip'] = df['precip'].astype(int)
        ds = df.drop('precip', axis=1)
        ds = ds.apply(pd.to_numeric, errors='coerce')
        ds.fillna(0, inplace=True)
        idx = self._add_to_cache(ds.values, file_path)
        file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)
        self.data_info[file_idx + idx]['cache_idx'] = idx

        label = df['precip'].values
        idx = self._add_to_cache(label, file_path)
        file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)
        self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]
