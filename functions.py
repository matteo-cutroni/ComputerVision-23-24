from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np

def assign_team(frame, results, model, player_color, team_colors, cap):
    for i, xywh in enumerate(results[0].boxes.xywh):
        # operazione da fare solo per le persone
        if model.names[int(results[0].boxes.cls[i])] != 'person':

            print(f'\n\n Ball Detected?\n tracked a {model.names[int(results[0].boxes.cls[i])]} at frame: {cap.get(cv.CAP_PROP_POS_FRAMES)}\n\n')
            continue
        # il colore è assegnato solo a nuovi id
        if int(results[0].boxes.id[i]) in player_color:
            continue

        x, y, w, h = [int(xywh[j]) for j in range(4)]
        left = int(x-w/2)
        top = int(y-h/2)
        crop = results[0].orig_img[top:top+h, left:left+w]

        # prendo solo parte alta (per prendere il colore della maglietta)
        top_crop = crop[:int(crop.shape[0]/2), :]

        # modifico in array 2d diviso per colori di pixel
        flat_crop = top_crop.reshape(-1, 3)

        kmeans_player = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=42)
        kmeans_player.fit(flat_crop)

        labels = kmeans_player.labels_

        # il colore della maglietta è quello della label meno presente
        tot = sum(labels)

        if tot > flat_crop.shape[0]/2:
            player_color[int(results[0].boxes.id[i])] = kmeans_player.cluster_centers_[0]
        else:
            player_color[int(results[0].boxes.id[i])] = kmeans_player.cluster_centers_[1]


    # al primo frame vengono assegnati i colori delle due squadre
    if len(team_colors) == 0:

        kmeans_team = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=42)
        kmeans_team.fit(list(player_color.values()))

        team_colors = kmeans_team.cluster_centers_


    for i, xywh in enumerate(results[0].boxes.xywh):
        # operazione da fare solo per le persone
        if model.names[int(results[0].boxes.cls[i])] != 'person':
            continue          

        x, y, w, h = [int(xywh[j]) for j in range(4)]
        left = int(x-w/2)
        top = int(y-h/2)

        # colore assegnato solo a nuovi giocatori tracciati
        if player_color[int(results[0].boxes.id[i])] not in team_colors:

            # assegnato al giocatore il colore della squadra più vicino a quello della sua maglietta
            distances = [np.linalg.norm(player_color[int(results[0].boxes.id[i])] - 1.5*color) for color in team_colors]
            player_color[int(results[0].boxes.id[i])] = team_colors[np.argmin(distances)]

        cv.rectangle(frame, (left, top), (left+w, top+h), player_color[int(results[0].boxes.id[i])], 2)
        cv.putText(frame, f'ID: {int(results[0].boxes.id[i])}', (left, top+5), cv.FONT_HERSHEY_COMPLEX, 1, player_color[int(results[0].boxes.id[i])], 2)



def top_view():
    return