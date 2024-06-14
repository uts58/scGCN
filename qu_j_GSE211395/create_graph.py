import argparse
import os

import networkx as nx
from torch_geometric.utils import from_networkx

from config import *


def create_common_graph(resolution=resolution):
    nodes_with_attributes = []
    # edge_list = []

    for size, chrom_name in zip(chrom_sizes_['size'], chrom_sizes_['chrom_name']):
        chr_bin = size // resolution
        for bins in range(chr_bin + 1):
            nodes_with_attributes.append(
                (f"{chrom_name}_{bins}")
            )

        # edges = [(f"{chrom_name}_{i}", f"{chrom_name}_{i + 1}") for i in range(chr_bin)]
        # edge_list.extend(edges)

    g = nx.Graph()
    g.add_nodes_from(nodes_with_attributes)
    # g.add_edges_from(edge_list)

    return g


def construct_final_graph(common_g, file_base_name_: str, resolution=resolution):
    # itertuples structure: Pandas(chr1='chr1', pos1=3017794, chr2='chr1', pos2=4261652)
    column_names = [
        'readID', 'chr1', 'pos1', 'Strand_1', 'chr2', 'pos2', 'Strand_2',
        'Distance', 'HIC_ID_1', 'HIC_ID_2', 'Quality_1', 'Quality_2'
    ]

    df_data = pd.read_csv(file_base_name_,
                          delim_whitespace=True,
                          comment='#',
                          names=column_names)

    df_data = df_data[['chr1', 'pos1', 'chr2', 'pos2']]
    df_data = df_data.loc[
        df_data['chr1'].isin(config_['chrom_list']) & df_data['chr2'].isin(config_['chrom_list'])
        ].copy()
    edge_list = [(f"{row[0]}_{row[1] // resolution}",
                  f"{row[2]}_{row[3] // resolution}")
                 for row in df_data.itertuples(index=False)]
    common_g.add_edges_from(edge_list)

    return common_g


chrs = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5',
    'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
    'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
    'chr16', 'chr17', 'chr18', 'chr19',
    # 'chr20', 'chr21', 'chr22',  #mouse doesn't have these
    'chrX'
]

parser = argparse.ArgumentParser(description="Process some chromosomes.")
parser.add_argument('chrs', metavar='N', type=str, nargs='+', help='List of chromosomes')
args = parser.parse_args()
chrs = args.chrs

cell_vs_files = get_data_files('.allValidPairs.txt')
print('list of pair files generated')

for chr_ in chrs:
    print(f'Starting {chr_}')
    config_['chrom_list'] = [chr_]
    dir_ = f'{config_["parent_dir"]}/graphs/_{chr_}'
    os.makedirs(dir_, exist_ok=True)

    chrom_sizes_ = get_mouse_chr_sizes()

    for items in cell_vs_files:
        file_base_name = cell_vs_files[items]
        cell_name = items

        common_graph = create_common_graph()
        print(f'Common_graph: Number of edges:{common_graph.number_of_edges()}, number of nodes:{common_graph.number_of_nodes()}')

        final_graph = construct_final_graph(common_graph, file_base_name)
        print(f'Final_graph: {cell_name}: Number of edges:{final_graph.number_of_edges()}, number of nodes:{final_graph.number_of_nodes()}')

        try:
            data = from_networkx(final_graph)
            # data.x = data.x.view(-1, 1)
            pickle.dump(data, open(f'{dir_}/{cell_name}.pkl', 'wb'))
            print(f'Saved to {dir_}/{cell_name}.pkl')
        except ValueError:
            print(f'{cell_name} data dump failed')
