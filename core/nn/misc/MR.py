
# See:
# - Learning embeddings for speaker clustering based on voice equality
# - Unfolding Speaker Clustering Potential: A Biomimetic Approach
# - http://users.uoi.gr/cs01702/MargaritaKotti/MypublicationsPDFs/J3.pdf

# Yanick:
# Mer luegt eigentlich eifach d reinheit vode cluster a, also d MR startet bi 1.0 und gaht jedes mal wenn zwei datepünkt
# (vom glieche sprecher) gmergt werded abe. Gmergti datepünkt gäbed wieder en neue datepunkt. Wenn zb 2 richtig gmergt
# werded und denn im nächste schritt de resultierti datepunkt mit eme datepunkt vomene andere sprecher gmergt wird werded
#     alli datepünkt i dem neue datepunkt (cluster) als falsch bewertet
# => nur reini cluster werded ade entsprechende klass zuegordnet
#  Meine Annahme:
# [09:43, 10.10.2017] +41 78 664 79 21: Oke, merci:)
#
# Demfall isch d'formla so?:
# sum_e_i = 0
# foreach cluster c:
#    e_i = 0
#    if not is_pure(cluster):
#        e_i += len(cluster)
#    sum_e_i += e_i
# MR = sum_e_i / ANZAHL_ELEMENTE_DIE_INSGESAMT_VORHANDEN_SIND
# (Wobi min Code wör e MR vu 0.0 usgeh wenn jede Datepunkt i en eigene Cluster wör gsetzt werde)

# Wie kann man MR berechnen, falls die Anzahl Cluster falsch geschätzt wurde?

def misclassification_rate(y_true, y_pred):

    pass
