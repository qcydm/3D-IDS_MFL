import ipaddress
import os
import time
from abc import ABC
from bisect import bisect_left

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, TemporalData
from z3 import Optimize, RealVector, Sum, And, sat


# 定义自己的数据集类
class ToNDataset(Dataset, ABC):
    def __init__(self, root='./data/', transform=None, pre_transform=None):
        super(ToNDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return 'CIC-ToN-IoT.csv'

    @property
    def processed_file_names(self):
        return 'CIC-ToN-IoT.pt'

    # def download(self):
    #     # download_url('https://cloudstor.aarnet.edu.au/plus/s/hubG06ytIBXoCGi/download?path=%2F&files=CIC-ToN-IoT.csv&downloadStartSecret=jgc5y7o4cyf','./data')
    #     pass

    def process(self):
        DISENTANGLE = False
        if os.path.exists(f'./data/{self.processed_file_names}'):
            print(f'Path ./data/{self.processed_file_names} already existed.')
            return
        else:
            print("Preparing dataset CiC-ToN-IoT.")
            df = pd.read_csv(f'./data/{self.raw_file_names}')
            print("Read csv done.")
            src_matches = df['Src IP'].str.endswith(('.0', '.1', '.255'))
            dst_matches = df['Dst IP'].str.endswith(('.0', '.1', '.255'))
            df = df.drop(columns=["Flow ID"])
            df.insert(loc=0, column='src', value=0)
            df["src"] = df.apply(lambda x: self.addr2num(x['Src IP'], int(x['Src Port'])), axis=1)
            df.insert(loc=1, column='dst', value=0)
            df["dst"] = df.apply(lambda x: self.addr2num(x['Dst IP'], int(x['Dst Port'])), axis=1)
            df = df.drop(columns=['Src IP', 'Dst IP', 'Src Port', 'Dst Port'])
            temp_data = df.pop("Timestamp")
            df.insert(2, 'timestamp', temp_data)
            temp_data = df.pop("Label")
            df.insert(3, 'state_label', temp_data)
            attack = df['Attack']
            df.drop(columns=['Attack'], inplace=True)
            print(pd.Categorical(attack).categories)
            attack = pd.Categorical(attack).codes
            df.insert(4, 'attack', attack)
            opt = list(df.columns.values)[5:]
            for name in opt:
                print(name)
                M = df[name].max()
                m = df[name].min()
                df[name] = df[name].apply(lambda x: ((x - m) / (M - m)) if (M - m) != 0 else 0)
            print("MIN-MAX done.")
            df.insert(5, 'layer i', 0)
            df.loc[src_matches, 'layer_i'] = 1
            df.insert(6, 'layer j', 0)
            df.loc[dst_matches, 'layer_j'] = 1
            temp_data = df.pop('Flow Duration')
            df.insert(7, 'Flow Duration', temp_data)
            df['timestamp'] = df['timestamp'].apply(
                lambda x: int(time.mktime(time.strptime(x, "%d/%m/%Y %I:%M:%S %p"))))
            df['timestamp'] = df['timestamp'] - df['timestamp'].min()
            print("Convert time done.")
            src_set = df.src.values
            dst_set = df.dst.values
            node_set = set(src_set).union(set(dst_set))
            ordered_node_set = sorted(node_set)
            assert (len(ordered_node_set) == len(set(ordered_node_set)))  # 查重
            df["src"] = df["src"].apply(lambda x: bisect_left(ordered_node_set, x) + 1)
            df["dst"] = df["dst"].apply(lambda x: bisect_left(ordered_node_set, x) + 1)
            print("Almost done.")
            df.sort_values(by="timestamp", inplace=True, ascending=True)
            print("Sort done.")
            df.fillna(0, inplace=True)
            df['layer_i'].value_counts()
            df.to_csv(f'./data/temp-{self.raw_file_names}')

            attack_types = df['attack'].unique()
            sampled_df = pd.DataFrame()
            for attack_type in attack_types:
                filtered_rows = df[df['attack'] == attack_type]
                sample_size = int(
                    len(filtered_rows) if 0.05 * len(filtered_rows) <= 1000 else 0.05 * len(filtered_rows))
                random_sample = filtered_rows.sample(n=sample_size)
                sampled_df = pd.concat([sampled_df, random_sample])

            sampled_df = sampled_df.sort_values('timestamp')
            sampled_df.to_csv(f'./data/selected-{self.raw_file_names}')
            print(sampled_df['attack'].value_counts())
            df = sampled_df

            # df = df.head(400000)[200000:]
            # df = pd.read_csv(f'./data/selected-{self.raw_file_names}')
            print(pd.Categorical(df['attack']).categories)
            df['attack'] = pd.Categorical(df['attack']).codes
            print(dict(df['state_label'].value_counts()))
            df.fillna(0, inplace=True)
            src = torch.tensor(df['src'].values.tolist())
            dst = torch.tensor(df['dst'].values.tolist())
            src_layer = torch.tensor(df['layer i'].values.tolist())
            dst_layer = torch.tensor(df['layer j'].values.tolist())
            label = torch.tensor(df['state_label'].values.tolist())
            t = torch.tensor(df['timestamp'].values.tolist())
            print(t)
            attack = torch.tensor(df['attack'].values.tolist())
            dt = torch.tensor(df['Flow Duration'].values.tolist())
            sdf = df.iloc[:, 8:]
            if DISENTANGLE:
                sdf_mean = sdf.mean()
                select_index = sdf_mean.sort_values().index
                select_index = select_index[:int(len(select_index) / 10)]
                sdf = sdf.apply(self.disentangle, args=(select_index,), axis=1)
            msg = torch.tensor(sdf.values.tolist())
            events = TemporalData(
                src=src,
                dst=dst,
                src_layer=src_layer,
                dst_layer=dst_layer,
                t=t,
                dt=dt,
                msg=msg,
                label=label,
                attack=attack)
            torch.save(events, f"./data/{self.processed_file_names}")
            return

    def addr2num(self, ip, port):
        bin_ip = bin(int(ipaddress.IPv4Address(ip))).replace("0b", "").zfill(32)
        bin_port = bin(port).replace('0b', '').zfill(16)
        id = bin_ip + bin_port
        id = int(id, 2)
        return id

    def solver(self, N):
        Wmin = 0
        Wmax = 1
        B = sum(N)
        s = Optimize()
        M = len(N)
        # print(M)
        W = RealVector('w', M)
        s.add(Sum([n * w for n in N for w in W]) < B)
        T = ''
        for i in range(0, M - 1):
            s.add(W[i] * N[i] <= W[i + 1] * N[i + 1])
            s.add(And(Wmin <= W[i], W[i] <= Wmax))
        for i in range(1, M - 1):
            s.add(2 * W[i] * N[i] <= W[i - 1] * N[i - 1] + W[i + 1] * N[i + 1])
            T = T + 2 * W[i] * N[i] - W[i - 1] * N[i - 1] - W[i + 1] * N[i + 1]
        s.maximize(W[M - 1] * N[M - 1] - W[0] * N[0] + T)

        if s.check() == sat:
            m = s.model()
            result = np.array(
                [float(m[y].as_decimal(10)[:-2]) if (len(m[y].as_decimal(10)) > 1) else float(m[y].as_decimal(10)) for y
                 in
                 W])

            return result

    def disentangle(self, N, select_axis):
        o = N[select_axis]
        t = self.solver(np.array(N[select_axis] + 0.01))
        if type(t) is np.ndarray and t.any() != None:
            N = N.replace(N[select_axis] + N[select_axis] * t)
            return N

    def get(self, idx=0):
        return torch.load(f'./data/{self.processed_file_names}')

    def len(self):
        pass

    def my_proess(self):
        file_path = 'data\dateset.csv'
        DISENTANGLE = True
        print("Preparing dataset CiC-BoT-IoT.")
        df = list(pd.read_csv('data\\test.csv',chunksize = 100000))[0]
        print("Read csv done.")
        src_matches = df['Src IP'].str.endswith(('.0', '.1', '.255'))
        dst_matches = df['Dst IP'].str.endswith(('.0', '.1', '.255'))
        df = df.drop(columns=["Flow ID"])
        df.insert(loc=0, column='src', value=0)
        df["src"] = df.apply(lambda x: self.addr2num(x['Src IP'], int(x['Src Port'])), axis=1)
        df.insert(loc=1, column='dst', value=0)
        df["dst"] = df.apply(lambda x: self.addr2num(x['Dst IP'], int(x['Dst Port'])), axis=1)
        df = df.drop(columns=['Src IP', 'Dst IP', 'Src Port', 'Dst Port'])
        temp_data = df.pop("Timestamp")
        df.insert(2, 'timestamp', temp_data)
        temp_data = df.pop("Label")
        df.insert(3, 'state_label', temp_data)
        attack = df['Attack']
        df.drop(columns=['Attack'], inplace=True)
        print(pd.Categorical(attack).categories)
        attack = pd.Categorical(attack).codes
        df.insert(4, 'attack', attack)
        opt = list(df.columns.values)[5:]
        for name in opt:
            print(name)
            M = df[name].max()
            m = df[name].min()
            df[name] = df[name].apply(lambda x: ((x - m) / (M - m)) if (M - m) != 0 else 0)
        print("MIN-MAX done.")
        df.insert(5, 'layer i', 0)
        df.loc[src_matches, 'layer_i'] = 1
        df.insert(6, 'layer j', 0)
        df.loc[dst_matches, 'layer_j'] = 1
        temp_data = df.pop('Flow Duration')
        df.insert(7, 'Flow Duration', temp_data)
        df['timestamp'] = df['timestamp'].apply(
            lambda x: int(time.mktime(time.strptime(x, "%d/%m/%Y %I:%M:%S %p"))))
        df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        print("Convert time done.")
        src_set = df.src.values
        dst_set = df.dst.values
        node_set = set(src_set).union(set(dst_set))
        ordered_node_set = sorted(node_set)
        assert (len(ordered_node_set) == len(set(ordered_node_set)))  # 查重
        df["src"] = df["src"].apply(lambda x: bisect_left(ordered_node_set, x) + 1)
        df["dst"] = df["dst"].apply(lambda x: bisect_left(ordered_node_set, x) + 1)
        print("Almost done.")
        df.sort_values(by="timestamp", inplace=True, ascending=True)
        print("Sort done.")
        df.fillna(0, inplace=True)
        df['layer_i'].value_counts()
        df.to_csv(f'./data/temp-tdest.csv')

        attack_types = df['attack'].unique()
        sampled_df = pd.DataFrame()
        for attack_type in attack_types:
            filtered_rows = df[df['attack'] == attack_type]
            sample_size = int(
                len(filtered_rows) if 0.05 * len(filtered_rows) <= 1000 else 0.05 * len(filtered_rows))
            random_sample = filtered_rows.sample(n=sample_size)
            sampled_df = pd.concat([sampled_df, random_sample])

        sampled_df = sampled_df.sort_values('timestamp')
        sampled_df.to_csv(f'./data/selected-test.csv')
        print(sampled_df['attack'].value_counts())
        df = sampled_df

        # df = df.head(400000)[200000:]
        # df = pd.read_csv(f'./data/selected-{self.raw_file_names}')
        print(pd.Categorical(df['attack']).categories)
        df['attack'] = pd.Categorical(df['attack']).codes
        print(dict(df['state_label'].value_counts()))
        df.fillna(0, inplace=True)
        src = torch.tensor(df['src'].values.tolist())
        dst = torch.tensor(df['dst'].values.tolist())
        src_layer = torch.tensor(df['layer i'].values.tolist())
        dst_layer = torch.tensor(df['layer j'].values.tolist())
        label = torch.tensor(df['state_label'].values.tolist())
        t = torch.tensor(df['timestamp'].values.tolist())
        attack = torch.tensor(df['attack'].values.tolist())
        dt = torch.tensor(df['Flow Duration'].values.tolist())
        sdf = df.iloc[:, 8:]
        if DISENTANGLE:
            sdf_mean = sdf.mean()
            select_index = sdf_mean.sort_values().index
            select_index = select_index[:int(len(select_index) / 10)]
            s = sdf.T
            sdf = sdf.apply(self.disentangle, args=(select_index,), axis=1)
            # print(sdf.T)
        msg = torch.tensor(sdf.values.tolist())
        events = TemporalData(
            src=src,
            dst=dst,
            src_layer=src_layer,
            dst_layer=dst_layer,
            t=t,
            dt=dt,
            msg=msg,
            label=label,
            attack=attack)
        torch.save(events, 'data\\CIC-BoT-IoT.pt')
        return


def main(self=ToNDataset()):
    # self.my_proess()
    self.process()


if __name__ == '__main__':
    main()
# dataset = ToNDataset()
