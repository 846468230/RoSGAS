from sklearn.cluster import  KMeans
from sklearn.metrics import homogeneity_score

def cul_hom_score(aggregated_features,labels):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(aggregated_features)
    pre_labels = kmeans.predict(aggregated_features)
    hom_score = homogeneity_score(labels, pre_labels)
    print(f"homogeneity score is: {homogeneity_score(labels, pre_labels)}")
    return hom_score