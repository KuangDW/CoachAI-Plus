# 轉換area
def coord2area(point_x, point_y, Real):
    if Real:
        point_x = point_x / 177.5
        point_y = point_y / 240
    mistake_landing_area = 33

    point_x = (point_x * (355/2)) + (355/2)
    point_y = (point_y * 240) + 240

    area1 = [[50,150],[104,204],1]
    area2 = [[104,150],[177.5,204],2]
    area3 = [[177.5,150],[251,204],3]
    area4 = [[251,150],[305,204],4]
    row1 = [area1, area2, area3, area4]

    area5 = [[50,204],[104,258],5]
    area6 = [[104,204],[177.5,258],6]
    area7 = [[177.5,204],[251,258],7]
    area8 = [[251,204],[305,258],8]
    row2 = [area5, area6, area7, area8]

    area9 = [[50,258],[104,312],9]
    area10 = [[104,258],[177.5,312],10]
    area11 = [[177.5,258],[251,312],11]
    area12 = [[251,258],[305,312],12]
    row3 = [area9, area10, area11, area12]
    
    area13 = [[50,312],[104, 366],13]
    area14 = [[104,312],[177.5,366],14]
    area15 = [[177.5,312],[251,366],15]
    area16 = [[251,312],[305,366],16]
    row4 = [area13, area14, area15, area16]

    area17 = [[50,366],[104,423],17]
    area18 = [[104,366],[177.5,423],18]
    area19 = [[177.5,366],[251,423],19]
    area20 = [[251,366],[305,423],20]
    row5 = [area17, area18, area19, area20]

    area21 = [[50,423],[104,480],21]
    area22 = [[104,423],[177.5,480],22]
    area23 = [[177.5,423],[251,480],23]
    area24 = [[251,423],[305,480],24]
    row6 = [area21, area22, area23, area24]

    area25 = [[305,366],[355,480],25]
    area26 = [[305,204],[355,366],26]
    area27 = [[305,0],[355,204],27]
    area28 = [[177.5,0],[305,150],28]
    row7 = [area25, area26, area27, area28]

    area29 = [[0,366],[50,480],29]
    area30 = [[0,204],[50,366],30]
    area31 = [[0,0],[50,204],31]
    area32 = [[50,0],[177.5,150],32]
    row8 = [area29, area30, area31, area32]

    check_area_list = row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8
    hit_area = mistake_landing_area
    for check_area in check_area_list:
        if point_x >= check_area[0][0] and point_y >= check_area[0][1] and point_x <= check_area[1][0] and point_y <= check_area[1][1]:
            hit_area = check_area[2]
    return hit_area