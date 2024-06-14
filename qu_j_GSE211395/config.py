import glob
import pickle
from collections import OrderedDict

import pandas as pd

config_ = {
    'config_name': 'uts et. al',
    'parent_dir': "/mmfs1/scratch/utsha.saha/mouse_data/data/not_using/qu_j_GSE211395_serum_2i/",
    'pairs_data_dir': "/mmfs1/scratch/utsha.saha/mouse_data/data/not_using/qu_j_GSE211395_serum_2i/hic/",
    'chrom_list': [
        'chr1', 'chr2', 'chr3', 'chr4', 'chr5',
        'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
        'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
        'chr16', 'chr17', 'chr18', 'chr19',
        # 'chr20', 'chr21', 'chr22',  #mouse doesn't have these
        'chrX'
    ],
    'rna_umicount_list': [
        "/mmfs1/scratch/utsha.saha/mouse_data/data/not_using/qu_j_GSE211395_serum_2i/scCARE-seq_RNA_raw_gene_counts.txt"
    ],
    'chr_gene_mapping_gencode': "/mmfs1/scratch/utsha.saha/mouse_data/data/chr_gene_mapping/gencode.vM25.annotation.gtf",
    'chr_gene_mapping_ncbi': "/mmfs1/scratch/utsha.saha/mouse_data/data/chr_gene_mapping/ncbi_grmc38_p6_annotation.gtf",
    'mouse_chr_sizes': "/mmfs1/scratch/utsha.saha/mouse_data/data/not_using/qu_j_GSE211395_serum_2i/mouse_chr_size.chr",
    'resolution': 50000
}

resolution = config_['resolution']


def create_chr_gene_mapping() -> dict:
    def process_csv(file_path: str, pattern: str, col_name: str) -> pd.DataFrame:
        df = pd.read_csv(
            file_path,
            sep='\t',
            comment='#',
            names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        )
        df = df[df['feature'] == 'gene'].copy()
        df[col_name] = df['attribute'].str.extract(pattern)
        return df

    def combine_synonyms(row: pd.Series) -> list:
        if isinstance(row['gene_synonyms'], list):
            return list(set(row['gene_synonyms'] + [row['gene_name']]))
        return [row['gene_name']]

    def create_mapping(df: pd.DataFrame) -> dict:
        mapping = {}
        for _, row in df.iterrows():
            for gene_synonym in row['gene_synonyms']:
                mapping[gene_synonym] = {
                    'original_gene_name': row['gene_name'],
                    'seqname': row['seqname'],
                    'start': row['start'],
                    'end': row['end'],
                    'bin': (row['start'] + row['end']) // 2 // resolution
                }
        return mapping

    df_ncbi = process_csv(config_['chr_gene_mapping_ncbi'], r'gene "([^"]+)"', 'gene_name')
    df_ncbi['gene_synonyms'] = df_ncbi['attribute'].str.findall(r'gene_synonym "([^"]+)"')
    df_ncbi = df_ncbi[['gene_name', 'gene_synonyms']]

    df_gencode = process_csv(config_['chr_gene_mapping_gencode'], r'gene_name "([^"]+)"', 'gene_name')
    df_gencode = df_gencode[['seqname', 'start', 'end', 'gene_name']]

    df_merged = pd.merge(df_gencode, df_ncbi, on='gene_name', how='left')
    df_merged['gene_synonyms'] = df_merged.apply(combine_synonyms, axis=1)
    df_filtered = df_merged[df_merged['seqname'].isin(config_['chrom_list'])].copy()
    return create_mapping(df_filtered)


def get_data_files(file_extension: str) -> dict:
    data_dir = config_['pairs_data_dir']
    cell_vs_files = {}

    for i in glob.glob(f"{data_dir}/*{file_extension}"):
        file_name = i
        cell_name = i.split('/')[-1].replace(file_extension, "")
        cell_vs_files[cell_name] = file_name

    return cell_vs_files


def get_mouse_chr_sizes():
    chrom_sizes = pd.read_csv(config_['mouse_chr_sizes'], sep='\t')
    chrom_sizes = chrom_sizes.loc[chrom_sizes['chrom_name'].isin(config_['chrom_list'])].copy()
    return chrom_sizes


def load_graph_data(dir_) -> dict:
    graph_dict = OrderedDict()
    dir_ = f'{dir_}/*.pkl'
    print(dir_)
    for files in glob.glob(dir_):
        graph_data = pickle.load(open(files, 'rb'))
        graph_data.edge_index = graph_data.edge_index.long()
        name = files.split('/')[-1].replace('.pkl', '')
        graph_dict[name] = graph_data

    return graph_dict
