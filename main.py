import pickle

import networkx as nx
from torch_geometric.utils import from_networkx

from config import *

df_rna_umicount = [pd.read_csv(rna_file, sep='\t') for rna_file in config_['rna_umicount_list']]
print('umicount files loaded')

cell_vs_files = get_data_files()
print('list of pair files generated')


def get_gene_umicount(cell_name, chrom_name_list=config_['chrom_list'], df_list=df_rna_umicount,
                      gene_data=chr_gene_mapping):
    to_return_main = {}

    for chrom_name in chrom_name_list:
        to_return = {}

        # Create a set of gene names for the specified chromosome
        relevant_genes = {gene for gene, data in gene_data.items() if data['seqname'] == chrom_name}

        for df in df_list:
            if cell_name not in df.columns:
                continue

            # Filter rows with positive umicounts for relevant genes
            filtered_df = df[df['gene'].isin(relevant_genes) & (df[cell_name] > 0)]

            for gene_name, umicount in zip(filtered_df['gene'], filtered_df[cell_name]):
                bin = gene_data[gene_name]['bin']
                to_return[bin] = to_return.get(bin, 0) + umicount

        to_return_main[chrom_name] = to_return

    return to_return_main


def create_common_graph(file_base_name_: str, cell_name_: str, resolution=resolution):
    chrom_sizes_ = pd.read_csv(f'{file_base_name_}.chr', sep='\t')
    chrom_sizes_ = chrom_sizes_.loc[chrom_sizes_['chrom_name'].isin(config_['chrom_list'])].copy()

    nodes_with_attributes = []
    edge_list = []

    gene_umicount = get_gene_umicount(cell_name_)

    for size, chrom_name in zip(chrom_sizes_['size'], chrom_sizes_['chrom_name']):
        chr_bin = size // resolution
        for bins in range(chr_bin + 1):
            if bins in gene_umicount[chrom_name].keys():
                nodes_with_attributes.append(
                    (f"{chrom_name}_{bins}", {'x': gene_umicount[chrom_name][bins]})
                )
            else:
                nodes_with_attributes.append(
                    (f"{chrom_name}_{bins}", {'x': 0})
                )

        edges = [(f"{chrom_name}_{i}", f"{chrom_name}_{i + 1}") for i in range(chr_bin)]
        edge_list.extend(edges)

    g = nx.Graph()
    g.add_nodes_from(nodes_with_attributes)
    g.add_edges_from(edge_list)

    return g


def construct_final_graph(common_g, file_base_name_: str, resolution=resolution):
    # itertuples structure: Pandas(chr1='chr1', pos1=3017794, chr2='chr1', pos2=4261652)
    df_data = pd.read_csv(f'{file_base_name_}.pairs', sep='\t', comment='#',
                          names=['readID', 'chr1', 'pos1', 'chr2', 'pos2', 'strand1', 'strand2', 'phase0', 'phase1'])
    df_data = df_data[['chr1', 'pos1', 'chr2', 'pos2']]
    df_data = df_data.loc[
        (df_data['chr1'].isin(config_['chrom_list'])) & (df_data['chr1'].isin(config_['chrom_list']))].copy()

    edge_list = [(f"{row[0]}_{row[1] // resolution}",
                  f"{row[2]}_{row[3] // resolution}")
                 for row in df_data.itertuples(index=False)]
    common_g.add_edges_from(edge_list)

    return common_g


for items in cell_vs_files:
    file_base_name = cell_vs_files[items]
    cell_name = items

    common_graph = create_common_graph(file_base_name, cell_name)
    print(
        f'Common_graph: {file_base_name}: Number of edges:{common_graph.number_of_edges()}, number of nodes:{common_graph.number_of_nodes()}')

    final_graph = construct_final_graph(common_graph, file_base_name)
    print(
        f'Final_graph: {file_base_name}: Number of edges:{final_graph.number_of_edges()}, number of nodes:{final_graph.number_of_nodes()}')

    try:
        data = from_networkx(final_graph)
        data.x = data.x.view(-1, 1)
        pickle.dump(data, open(f'{config_["graph_dir"]}/{cell_name}.pkl', 'wb'))
    except ValueError:
        print(f'{cell_name} data dump failed')
