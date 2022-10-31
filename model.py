import numpy as np
import pandas as pd
from netpixi.integration.gt import *
from regression.integration.gt import *
import math
from graph_tool import centrality, clustering

# Caminho para as pastas de read e save"
dir_save = "networks/"

# Nome do dataset
file_save_name = "vereadores"

# Extensões dos arquivos 
file_save_extension = ".net.gz"

##DATA
data_dir = "data/"

class Model:
    
    def __init__(self, df, nm, centros, y1=2013, y2=2020):
        self.df = df
        self.nm = nm
        self.centros = centros
        self.y1 = y1
        self.y2 = y2
        self.frame = str(self.y1) + "_" + str(self.y2)
        self.path_save = dir_save + file_save_name + "_" + self.frame + file_save_extension
        self.betweenness_file = dir_save + file_save_name + "_betweenness" + "_" + self.frame + file_save_extension
        self.clustering_file = dir_save + file_save_name + "_clustering" + "_" + self.frame + file_save_extension
        self.data_networks_path = data_dir + file_save_name + "_" + self.frame + ".csv"
    
    def drop_nulls(self):
        self.df = self.df[self.df['vereador'].notna() &
                          self.df['id_parlamentar'].notna() &
                          self.df['voto'].notna() &
                          self.df['id_votacao'].notna() &
                          self.df['resultado'].notna()]
    
    def weight_votes(self):
        self.df.loc[(self.df['voto'] == 'Sim') | 
                    (self.df['voto'] == 'Marco A Cunha (PSD)') | 
                    (self.df['voto'] == 'José Américo (PT)'), 
                    'voto'] = '2'
        self.df.loc[(self.df['voto'] == 'Não') | 
                    (self.df['voto'] == 'Souza Santos (PSD)') | 
                    (self.df['voto'] == 'Antônio Vespoli (PSOL)'), 
                    'voto'] = '0'
        self.df.loc[self.df['voto'] == 'Abstenção', 'voto'] = '1'
    
        df_clear_idx = list(x.isdigit() for x in self.df['voto'])
        self.df = self.df[df_clear_idx]
        self.df['voto'] = self.df.apply(lambda row: int(row['voto']) - 1, axis=1)
        
        return self.df
    
    def weight_results(self):
        not_results = [
            "Pendente de Votação",
            "Falta de Quórum",
            "PREJUDICADO POR FALTA DE QUÓRU",
            "PENDENTE DE VOTAÇÃO",
            "Prejudicado- Falta de Quórum",
            "Prejudicado - falta de quorum",
            "PREJUDICADO POR FALTA QUÓRUM",
            "Retirado",
        ]
        filter_results = list(x not in not_results for x in self.df['resultado'])
        self.df = self.df[filter_results]
        self.df.loc[
            (self.df['resultado'] == 'Aprovado') |
            (self.df['resultado'] == 'Eleito') |
            (self.df['resultado'] == 'APROVADO') |
            (self.df['resultado'] == 'Eleito José Américo (PT)') |
            (self.df['resultado'] == 'Eleito Marco A Cunha (PSD)'),
            'resultado'
        ] = 1
        self.df.loc[self.df['resultado'] == 'Reprovado', 'resultado'] = -1
        self.df.loc[self.df['resultado'] == 'Rejeitado', 'resultado'] = -1
        
        return self.df
    
    def framing(self):
        self.df['ano'] = self.df.apply(lambda row: int(row['data'][-4:]), axis=1)
        self.df = self.df[
            (self.df['ano'] >= self.y1) &
            (self.df['ano'] <= self.y2)
        ]
        return self.df
    
    def convergent_ids(self):
        cte = int(1e5)
        self.df['id_votacao'] = self.df['id_votacao'] + cte
    
    def parlament_votes(self):
        self.vereadores = np.unique(self.df['id_parlamentar'])
        self.n_vereadores = len(self.vereadores)
        self.vereadores_votos = []
        for i in range(self.n_vereadores):
            votes = (
                self.df[self.df['id_parlamentar'] == self.vereadores[i]]
                [['id_parlamentar', 'id_votacao', 'voto', 'resultado']]
            )
            votes['success'] = votes.apply(lambda row: 1 if row['voto'] == row['resultado'] else 0, axis=1)
            self.vereadores_votos.append(votes)
    
    def cut_point(self):
        self.vereadores_corr = np.zeros(shape=(self.n_vereadores, self.n_vereadores))
        list_vereadores_corr = []

        for i in range(self.n_vereadores):
            v1 = self.vereadores_votos[i]
            for j in range(i+1, self.n_vereadores - 1):
                v2 = self.vereadores_votos[j]
                corrs = (v1.merge(v2, how='inner', on='id_votacao')[['voto_x', 'voto_y']]
                         .corr(numeric_only = False))
                corr = corrs['voto_x']['voto_y']
                if math.isnan(corr):
                    corr = 0
                self.vereadores_corr[i][j] = corr
                list_vereadores_corr.append(corr)

        self.cut_point = np.median(list_vereadores_corr)
        return self.cut_point
    
    def network_create(self):
        self.g = Graph(directed=False)
    
        for i in range(self.n_vereadores):
            self.g.add_vertex(int(self.vereadores[i]))

        for i in range(self.n_vereadores):    
            for j in range(i+1, self.n_vereadores - 1):
                if self.vereadores_corr[i][j] > self.cut_point:
                    self.g.add_edge(
                        int(self.vereadores[i]),
                        int(self.vereadores[j])
                    )

        gt_save(self.g, self.path_save)
        
    def bet(self):
        g_betweenness = gt_load(self.path_save)
        bc, _ = centrality.betweenness(g_betweenness)
        g_betweenness.add_vp('betweenness', bc)
        gt_save(g_betweenness, self.betweenness_file)
        self.bet = gt_data(g_betweenness)
        return self.bet
    
    def bet_log(self):
        self.bet = self.bet[self.bet['betweenness'] > 0]
        self.bet['betweenness_log'] = self.bet.apply(lambda row: np.log10(row['betweenness']), axis=1)
        
    def clus(self):
        g_cluster = gt_load(self.path_save)
        lc = clustering.local_clustering(g_cluster)
        g_cluster.add_vp('clustering', lc)
        gt_save(g_cluster, self.clustering_file)
        self.clus = gt_data(g_cluster)
        return self.clus
    
    def rep(self):
        df_parlamentar = self.df.drop_duplicates(subset = "id_parlamentar").copy()
        df_tam = (
            df_parlamentar[["partido", "id_parlamentar"]]
            .groupby("partido").count()
            .rename(columns = {"id_parlamentar": "parlamentares"})
        )

        total_parlamentares = df_tam["parlamentares"].sum()
        df_tam['parlamentares'] = df_tam.apply(lambda row: row['parlamentares']/total_parlamentares, axis=1)

        self.rep = (
            self.df.merge(df_tam, on = "partido", how = "inner")[["id_parlamentar","parlamentares", "partido"]]
            .drop_duplicates("id_parlamentar")
            .rename(columns = {"id_parlamentar": "id"})
        )
        self.rep['extreme_party'] = self.rep.apply(lambda row : 1 if row['partido'] not in self.centros else 0, axis=1)
        
        return self.rep
    
    def suc(self):
        success_list = list()
        for i in range(self.n_vereadores):
            vereador_suc = (self.vereadores_votos[i]
                            [['id_parlamentar', 'success']]
                            .groupby('id_parlamentar')['success']
                            .agg(['sum','count'])
                            .reset_index())
            success_list.append((vereador_suc['id_parlamentar'][0],
                                 vereador_suc['sum'][0]/vereador_suc['count'][0]))
        self.suc = pd.DataFrame(success_list, columns=['id', 'success'])
        return self.suc
    
    def gen(self):
        self.df['first_name'] = self.df.apply(lambda row : row['vereador'].split(' ')[0]
                                              .replace('Á', 'A')
                                              .replace('É', 'E')
                                              .replace('Í', 'I')
                                              .replace('Ã', 'A')
                                              .replace('Â', 'A'),
                                              axis=1)
        self.df = (self.df.merge(self.nm[['first_name', 'classification']], on='first_name', how='left')
                   .rename(columns = {"classification": "gender"}))
        self.df.loc[self.df['first_name'] == "SONINHA", 'gender'] = 'F'
        self.df.loc[self.df['gender'].isnull(), 'gender'] = 'M'
        self.df['gender'] = self.df.apply(lambda row : 1 if row['gender'] == 'F' else 0, axis=1)
        self.gen = (self.df.drop_duplicates(subset = "id_parlamentar")
                    [['id_parlamentar', 'gender']]
                    .rename(columns = {"id_parlamentar": "id"}))
        return self.gen
    
    def var(self):
        self.var = (
            self.bet
            .merge(self.clus, on = 'id', how='inner')
            .merge(self.rep, on="id", how = "inner")
            .merge(self.suc, on="id", how = "inner")
            .merge(self.gen, on="id", how = "inner")
        )
        self.var.to_csv(self.data_networks_path)
        return self.var
    
    def do_all(self):
        self.drop_nulls()
        self.weight_votes()
        self.weight_results()
        self.framing()
        self.convergent_ids()
        self.parlament_votes()
        self.cut_point()
        self.network_create()
        self.bet()
        self.clus()
        self.rep()
        self.suc()
        self.gen()
        self.var()