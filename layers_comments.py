from torch import nn

class Attention(nn.Module):
    def __init__(self, input_features):                     
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, 128, 1, 1)                                                                                                             
        self.bn = nn.BatchNorm2d(128)                                         
        self.relu = nn.ReLU() 
        self.conv2 = nn.Conv2d(128, 1, 1, 1) 
        self.softplus = nn.Softplus()


    def forward(self, x, rec_feature_map=None):   
        input = x                                           
        x = self.conv1(x)                                  
        x = self.bn(x)                                         
        x = self.relu(x)                                   

        score = self.conv2(x)                                             
        prob = self.softplus(score)                        

        if rec_feature_map is None:                     
            rec_feature_map = input   
        
        rec_feature_map_norm = F.normalize(rec_feature_map, p=2, dim=1) 
        att = torch.mul(rec_feature_map_norm, prob)        
        feat = torch.mean(att, [2, 3])    
        # feat = tf.reduce_mean(tf.multiply(targets, prob), [1, 2], keepdims=False)                            
        # print(f"att:{feat.shape}")
        return feat, prob, att  
                       
#self.conv1 = nn.Conv2d(input_features, 128, 1, 1)       
# Per prima cosa abbiamo il passaggio attraverso un layer convoluzionale che riduce solo ed esclusivamente il numero di features.
# la riduzione delle features in un modulo di attenzione può portare a benefici come riduzione della complessità, focalizzazione 
# dell'attenzione su aspetti rilevanti, gestione del rumore e ottimizzazione delle risorse.

# self.bn = nn.BatchNorm2d(128) 
# Dopodiché viene applicata la Batch normalization che calcola media e varianza sull'intero batch, dopodiché lo centra e lo 
# normalizza in modo da mitigare problemi come l'instabilità del gradiente e l'effetto della covarianza tra le feature, che 
# possono influenzare negativamente l'addestramento della rete neurale. Inoltre, la normalizzazione del batch può contribuire 
# a rendere il modello meno sensibile alle variazioni nei dati di input e a migliorare la generalizzazione.

# self.relu = nn.ReLU() 
# L'applicazione di una funzione di attivazione ReLU (Rectified Linear Unit) dopo la Batch Normalization ha l'obiettivo di introdurre
# non linearità e rimuovere i valori negativi.

# self.conv2 = nn.Conv2d(128, 1, 1, 1) 
# self.softplus = nn.Softplus()
# Applicando la seconda convoluzione e la softplus, otteniamo a tutti gli effetti la mappa di attenzione.
# In particolare, la softplus viene utilizzata per generare una mappa di attenzione con valori positivi, mentre i valori negativi 
# rimangono bassi ma non sono completamente annullati. L'interpretazione e l'utilizzo dei valori negativi dipendono dal contesto 
# dell'applicazione specifica.   

# L'applicazione della prima convoluzione che porta le features da 256 a 128 sembrerebbe solo servire ad aiutare a concentrare 
# l'attenzione sulle caratteristiche più salienti o informative, riducendo al contempo la complessità computazionale, perché tanto
# alla fine la seconda convoluzione ridurrà nuovamente le features ad 1

# rec_feature_map_norm = F.normalize(rec_feature_map, p=2, dim=1) 
# Successivamente viene applicata la normalizzazione all'input (eveuntualmente ridotto dall'autoencoder) prima di essere utilizzata 
# per calcolare l'attenzione. Questo assicura che le caratteristiche siano normalizzate e che la loro importanza relativa venga calcolata 
# in base ai pesi dell'attenzione senza distorsioni dovute alle differenze di ampiezza tra le caratteristiche.

# att = torch.mul(rec_feature_map_norm, prob) 
# Esegue un prodotto elemento per elemento tra la mappa delle caratteristiche normalizzate rec_feature_map_norm e la mappa di attenzione
# L'obiettivo di questa operazione è quello di pesare le diverse aree dell'immagine o delle caratteristiche in base alla loro importanza 
# relativa, come indicato dalla mappa di attenzione. Le aree dell'immagine che hanno valori di attenzione più alti avranno un peso maggiore 
# nelle fasi successive del processo di attenzione, mentre le aree con valori di attenzione più bassi avranno un peso inferiore

# feat = torch.mean(att, [2, 3])   
# Si ottiene un tensore aggregato di dimensione ridotta rispetto alla mappa di attenzione originale. Questo tensore rappresenta l'aggregazione 
# delle informazioni di attenzione per ciascuna feature. Non avendo quindi più l'informazione spaziale, è come se per ogni feature avessi il suo
# valore di importanza, ottenuto tramite il prodotto con la mappa delle attenzioni  

class Autoencoder(nn.Module):
    def __init__(self, input_features, output_features):   
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, 1, 1)  
        self.conv2 = nn.Conv2d(output_features, input_features, 1, 1)  
        self.relu = nn.ReLU()                             

    def forward(self, x):                                  
        reduced_dim = self.conv1(x)                     
        x = self.conv2(reduced_dim)                    
        expanded_dim = self.relu(x)           
        return reduced_dim, expanded_dim
