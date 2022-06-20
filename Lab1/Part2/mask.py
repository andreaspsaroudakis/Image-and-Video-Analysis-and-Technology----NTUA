import numpy as np

def create_mask(resized_frame):

    #ouranos/dentra
    resized_frame [0:125,:,:] = [255,255,255]
    resized_frame [125:135, 75:413,:]= [255,255,255]
    resized_frame [135:140,75:300,:]=[255,255,255]
    resized_frame [125:145,35:100,:] = [255,255,255]
    resized_frame [125:145,0:35:] = [255,255,255]

    #aristera dentro
    resized_frame [315:360,0:75,:] = [255,255,255]
    resized_frame [250:315,0:65,:] = [255,255,255]
    resized_frame [239:250,0:55,:] = [255,255,255]
    resized_frame [250:330,0:75,:]= [255,255,255]
    resized_frame [290:330,0:93,:]= [255,255,255]
    resized_frame [330:,0:100,:] = [255,255,255]

    #deksia dentro
    resized_frame [135:250,628:,:] = [255,255,255]
    resized_frame [157:240,610:,:] = [255,255,255]
    resized_frame [180:220,600:,:] = [255,255,255]
    resized_frame [200:220,597:,:] = [255,255,255]
    resized_frame [220:233,603:,:] = [255,255,255]
    resized_frame [233:240,605:,:] = [255,255,255]
    resized_frame [240:244,607:,:] = [255,255,255]
    resized_frame [244:252,622:,:] = [255,255,255]

    #fws sto dromo
    resized_frame [257:261,555:581,:] = [255,255,255]
    resized_frame [257:264,555:581,:] = [255,255,255]
    resized_frame [261:268,585:605,:] = [255,255,255]
    resized_frame [266,576,:]= [255,255,255]
    resized_frame [201,532,:] = [255,255,255]

    #stop
    resized_frame[276:290,408:418]= [255,255,255]
    resized_frame[290:319,411:412]= [255,255,255]

    #fws sto parking
    resized_frame [201,532,:]=[255,255,255]
    resized_frame[261:267,570:585]= [255,255,255]

    #fws1
    resized_frame[145:180,244:247,:]=[255,255,255]
    resized_frame[180:217,243:246,:]=[255,255,255]

    #pinakida
    resized_frame[269:294,225:242,:]=[255,255,255]

    #prostateftika
    resized_frame[252:268,198:201]=[255,255,255]
    resized_frame[252:270,190:193]=[255,255,255]
    resized_frame[258:275,172:175]=[255,255,255]
    resized_frame[260:281,156:161]=[255,255,255]

    #kioski
    resized_frame[204:230,0:58,:]=[255,255,255]

    #fanari
    resized_frame[186:202,530:535,:]=[255,255,255]

    #####################################################################################################

    #ouranos/dentra
    mask = np.ones((resized_frame.shape[0],resized_frame.shape[1]),dtype=np.uint8)
    mask [0:125,:]= 0
    mask [125:135, 75:413]=0
    mask [135:140,75:300]=0
    mask [125:145,35:100]=0
    mask [125:145,0:35] = 0

    #aristera dentro
    mask [315:360,0:75] = 0
    mask [250:315,0:65] = 0
    mask [330:,0:100] = 0
    mask [239:250,0:55] = 0
    mask [250:330,0:75]= 0
    mask [290:330,0:93]= 0

    #deksia dentro
    mask [135:250,628:] = 0
    mask [157:240,610:] = 0
    mask [180:220,600:] = 0
    mask [200:220,597:] = 0
    mask [220:233,603:] = 0
    mask [233:240,605:] = 0
    mask [240:244,607:] = 0
    mask [244:252,622:] = 0

    #fws sto dromo
    mask [257:261,555:581] = 0
    mask [257:264,555:581] = 0
    mask [261:268,585:605] = 0
    mask [266,576]= 0
    mask [201,532] = 0

    #stop
    mask[276:290,408:418]= 0
    mask[290:319,411:413]= 0

    #fws sto parking
    mask [201,532]=0
    mask[261:267,570:585]= 0

    #fws1
    mask[145:180,244:247]=0
    mask[180:217,243:246]=0

    #pinakida
    mask[269:294,225:242]=0

    #prostateftika
    mask[252:268,198:201]=0
    mask[252:270,190:193]=0
    mask[258:275,172:175]=0
    mask[260:281,156:161]=0

    #kioski
    mask[204:230,0:58]=0

    #fanari
    mask[186:202,530:535]=0

    return mask, resized_frame