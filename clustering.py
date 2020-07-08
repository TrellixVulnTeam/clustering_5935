import sys

import pandas as pd
import io
from sklearn.cluster import DBSCAN
from text_tools import distance, similarity, distance2
import numpy as np


class IncrementalDBSCAN:

    def __init__(self, eps=5, min_samples=2):
        """
        Constructeur de la classe Incremental_DBSCAN.
        :param eps:  le rayon maximum qu'un élément doit être afin de formuler un cluster
        :param min_samples:  les échantillons minimum requis pour formuler un cluster
        Afin d'identifier les eps et min_samples optimaux, nous devons créer un KNN
        """
        self.dataset = pd.DataFrame(columns=['Type', 'Message'])
        self.labels = pd.DataFrame(columns=['Label'])
        self.final_dataset = pd.DataFrame(columns=['Type', 'Message', 'Label'])
        self.mean_core_elements = pd.DataFrame(columns=['Type', 'Message', 'Label'])
        self.eps = eps
        self.min_samples = min_samples
        self.largest_cluster = -1
        self.cluster_limits = 0
        self.largest_cluster_limits = 0

    def set_data(self, message):
        """
        Une fois que la connexion avec le RabbitMQ est terminée, un message est reçu.
        Cette fonction est utilisée pour recueillir le message du consommateur. Elle ajoute les données nouvellement arrivées au
        ensemble de données utilisé pour le regroupement.
        :param message:  Le message consommé par le RabbitMQ. Doit être un texte de 3 colonnes, séparées par des virgules.
        """
        # stocker le message collecté dans une dataframe de données temporaire
        # TODO separator !
        temp = pd.read_csv(io.StringIO(message), sep='|', header=None)
        # Split et ajouter directement

        temp.columns = ['Type', 'Message']
        self.dataset = self.dataset.append(temp, ignore_index=True)

    def batch_dbscan(self):
        """
        L'algorithme DBSCAN est tiré de la bibliothèque sklearn. Il est utilisé pour formuler les clusters la première fois.
        Sur la base des résultats de cet algorithme, l'algorithme DBSCAN incrémental
        """

        n = len(self.dataset)
        dataX = self.dataset.copy()
        X = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                temp = 10 - 10 * similarity(dataX.iloc[i]['Message'], dataX.iloc[j]['Message'])
                if temp < 0:
                    X[i][j] = 0
                else:
                    X[i][j] = temp
        # batch_dbscan = DBSCAN(eps=1.3, min_samples=3, metric='precomputed').fit(X)
        batch_dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed').fit(X)
        # recuperer le nombre de cluster creer
        n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        print("Nombre de cluster ", n_clusters_)
        # self.add_labels_to_dataset(batch_dbscan.get_cluster_labels())
        self.add_labels_to_dataset(batch_dbscan.labels_)

        self.final_dataset = self.final_dataset.astype(str)

    def add_labels_to_dataset(self, labels):
        """
        Cette fonction ajoute les étiquettes sur l'ensemble de données après que le lot DBSCAN ait été effectué
        :param labels: Le paramètre labels doit être une liste qui décrit le groupe de chaque élément.
        Si un élément est considéré comme une valeur aberrante, il doit être égal à -1
        """
        self.labels = pd.DataFrame(labels, columns=['Label'])
        self.final_dataset = pd.concat([self.dataset, self.labels], axis=1)

    def sort_dataset_based_on_labels(self):
        """
        Cette fonction trie l'ensemble des données en fonction de l'étiquette de chaque grappe.
        """
        # print(self.final_dataset)
        self.final_dataset = self.final_dataset.sort_values(by=['Label'])
        self.final_dataset = self.final_dataset.astype(str)

    def find_mean_core_element(self):
        """
        Cette fonction calcule la moyenne des éléments de base de chaque groupe.
        Remarque : elle ne calcule pas d'élément central moyen pour les valeurs aberrantes.
        """
        # Exclure les lignes étiquetées comme aberrantes
        self.mean_core_elements = self.final_dataset.loc[self.final_dataset['Label'] != -1]

    def calculate_min_distance_centroid(self):
        """
        Cette fonction identifie l'élément mean_core_element le plus proche de l'élément entrant
        qui n'a pas encore été ajoutée à un groupe ou considérée comme aberrante.
        La distance est calculée à l'aide de la fonction de distance telle qu'elle est décrite ci-dessus.

        :returns min_dist_index: s'il existe un groupe qui est le plus proche du nouvel élément d'entrée
        ou Aucun s'il n'y a pas encore de clusters.
        """
        min_dist = None
        min_dist_index = None

        # Vérifiez s'il y a des éléments dans le dataframe core_elements.
        # En d'autres termes, s'il y a des clusters créés par l'algorithme DBSCAN
        if not self.mean_core_elements.empty:
            for index, current_mean_core_element in self.mean_core_elements.iterrows():

                try:
                    tmp_dist = distance2(type_1=self.final_dataset.tail(n=1).iloc[0]['Type'],
                                         type_2=current_mean_core_element['Type'],
                                         message_1=self.final_dataset.tail(n=1).iloc[0]['Message'],
                                         message_2=current_mean_core_element['Message'])
                except:
                    # ICI
                    print("Unexpected error:", sys.exc_info()[0])
                    print("Element 1 type ", type(self.final_dataset.tail(n=1)))
                    print("Element 2 type ", type(current_mean_core_element))
                    tmp_dist = 0

                if min_dist is None:
                    min_dist = tmp_dist
                    min_dist_index = index
                elif tmp_dist < min_dist:
                    min_dist = tmp_dist
                    min_dist_index = index
            print('Minimum distance is: ', min_dist, ' at cluster ', min_dist_index)
            return min_dist_index
        else:
            return None

    def check_min_samples_in_eps_or_outlier(self, min_dist_index):
        """
        Cette fonction vérifie s'il y a au moins min_échantillons dans le rayon donné du nouveau
        élément d'entrée.
        S'il y a au moins min_samples, cet élément sera ajouté à la grappe et l'élément
        mean_core_element du cluster actuel doit être recalculé.
        Sinon, il y a deux options.
            1. Vérifier s'il y a au moins des valeurs aberrantes de min_samples dans le rayon donné afin de créer un nouveau
                de la grappe, ou
            2.  Considérez qu'il s'agit d'une nouvelle valeur aberrante

        :param min_dist_index: Il s'agit du paramètre qui contient les informations relatives à la
        mean_core_element à l'élément courant.
        """

        # N'utiliser que les éléments de la grappe la plus proche de l'élément de la nouvelle entrée
        new_element = self.final_dataset.tail(1)
        nearest_cluster_elements = self.final_dataset[self.final_dataset['Label'] == min_dist_index]
        min_samples_count = 0
        for index, cluster_element in nearest_cluster_elements.iterrows():
            if distance2(cluster_element['Type'], new_element['Type'], cluster_element['Message'],
                         new_element['Message']) <= self.eps:
                min_samples_count += 1

        if min_samples_count >= self.min_samples:

            self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = min_dist_index
            self.find_mean_core_element()
        else:
            outliers = self.final_dataset[self.final_dataset['Label'] == -1]
            min_outliers_count = 0
            new_cluster_elements = pd.DataFrame(columns=['Index'])
            # print("Outliers ", outliers.items()[0])
            for index, outlier in outliers.iterrows():
                # print("Outlier : ", type(outlier))
                # print("*****************************")
                # print("Type new_element : ", type(new_element.iloc[0]))
                # print("*****************************")
                # print("Message outlier ", outlier, "\nNew_element ", new_element.iloc[0])
                if distance2(outlier['Type'], new_element.iloc[0]['Type'], outlier['Message'],
                             new_element.iloc[0]['Message']) <= self.eps:
                    min_outliers_count += 1
                    new_cluster_elements = new_cluster_elements.append({"Index": index}, ignore_index=True)

            if min_outliers_count >= self.min_samples:
                print("Types of labels in final dataset ", type(self.final_dataset['Label']))
                print("Content in final dataset", self.final_dataset)
                print("Content in label ", self.final_dataset['Label'])
                new_cluster_number = self.fin_max_label() + 1
                for new_cluster_element in new_cluster_elements.iterrows():
                    self.final_dataset.loc[
                        self.final_dataset.index[int(new_cluster_element[1])], 'Label'] = new_cluster_number

                print("A new cluster is now formed out of already existing outliers.")

                # L'élément central moyen de la nouvelle grappe est calculé après la création de la grappe.
                self.find_mean_core_element()

            else:
                # Le nouvel élément est une valeur aberrante.
                # Il n'est pas assez proche de son plus proche pour pouvoir y être ajouté,
                # ni n'a assez de cas aberrants à proximité pour former un nouveau groupe.
                self.final_dataset.loc[self.final_dataset.index[-1], 'Label'] = -1

        print("le nouveau element dans la base de donnees est : \n", self.final_dataset.tail(1))

    def incremental_dbscan_(self):
        self.final_dataset = self.final_dataset.append({'Type': self.dataset.iloc[-1]['Type'],
                                                        'Message': self.dataset.iloc[-1]['Message'],
                                                        'Label': -1}, ignore_index=True)
        self.find_mean_core_element()
        min_distance_mean_core_element_index = self.calculate_min_distance_centroid()
        if min_distance_mean_core_element_index is not None:
            self.check_min_samples_in_eps_or_outlier(min_dist_index=min_distance_mean_core_element_index)
        self.largest_cluster = self.find_largest_cluster()
        self.find_cluster_limits()
        self.get_largest_cluster_limits()
        print("*******************************************************************************")
        print("Data SET : ")
        print(self.final_dataset)
        print("*******************************************************************************")

    def find_largest_cluster(self):
        """
        Cette fonction permet d'identifier le plus grand des groupes par rapport au nombre d'éléments de base.
        La plus grande grappe est celle qui contient le plus grand nombre d'éléments fondamentaux.

        :returns : le numéro de la plus grande grappe. Si -1 est renvoyé, alors il n'y a pas de grappes créées
        en premier lieu.
        """
        cluster_size = self.final_dataset.groupby('Label')['Label'].count()
        try:
            cluster_size = cluster_size.drop(labels=[-1])
        except ValueError:
            print("Le label -1 n'existe pas")
        largest_cluster = -1
        if not cluster_size.empty:
            largest_cluster = cluster_size.idxmax()
            print('''Le groupe qui comporte le plus d'éléments est le groupe Non''', cluster_size.idxmax())
            return largest_cluster
        else:
            print('Il n\'y a pas encore de groupes formés')
            return largest_cluster

    def find_cluster_limits(self):
        self.cluster_limits = self.final_dataset \
            .groupby(self.final_dataset['Label']) \
            .agg(['min', 'max'])
        print(self.cluster_limits)
        self.cluster_limits.to_json(r'jsons/all_cluster_limits.json')

    def get_largest_cluster_limits(self):
        self.largest_cluster_limits = self.cluster_limits.iloc[int(self.largest_cluster) + 1]
        self.largest_cluster_limits.to_json(r'jsons/largest_cluster_limits.json')
        print(self.largest_cluster_limits)

    def fin_max_label(self):
        max = -1
        for index, value in self.final_dataset['Label'].items():
            print("Index : ", index, "Value :", value)
            print("Index : ", index, "Value :", self.final_dataset['Label'].iloc[index])
            # print("Index : {index}, Value : {self.final_dataset[index]['Label']}")
            if int(self.final_dataset['Label'].iloc[index]) > max:
                max = int(self.final_dataset['Label'].iloc[index])
                print("Here new MAX", max)
        return max
