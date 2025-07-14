import numpy as np
import cartopy.crs as ccrs
from typing import Union
from .proj_info import proj_LC, proj_MERC

class model_info_2d(object):
    """
    用于创建模式网格, 并包含了相关信息, 提供了方便坐标与经纬度相互转换的工具
    基于Numpy和Cartopy.crs构建, 仅支持方形网格
    """
    def __init__(
            self,
            proj        : ccrs.PlateCarree  = None,
            nx          : int               = None,
            ny          : int               = None,
            dx          : float             = None,
            dy          : float             = None,
            lowerleft   : list              = None,
            nt          : int               = None,
            dt          : float             = None,
            var_list    : list              = None,
            type        : str               = None,
            globe       : ccrs.Globe        = None,
            debug       : int               = 0,
            center      : list              = None,
            rotate_deg  : Union[int, float] = 0,
            rotate_poi  : list              = None,
    ) -> None:
        
        """
        用于初始化网格, 如果不给定左下角经纬度坐标, 则默认投影坐标原点位置为网格
        中心, 并依据此建立网格
        必选参数:
            proj    : 目标网格所在的投影, 是cartopy.crs类
            nx      : x方向网格个数
            ny      : y方向网格个数
            dx      : x方向网格距离(在目标网格投影下, 例如兰伯特是米, 等经纬是度)
            dy      : y方向网格距离
        可选参数:
            lowerleft_lonlat    : 左下角坐标(经纬度)
            nt                  : 每个模式输出文件的时间段个数
            dt                  : 每个模式输出文件的时间间隔(小时)
            var_list            : 模式包含的变量列表
            type                : 模式的类型(只是一个标记)
            globe               : 地球形状设定
            debug               : 设置打印的信息
        更新记录:
            2022-08-20 22:08:27 Sola 编写源代码
            2022-08-20 22:08:33 Sola 添加注释
            2022-08-21 11:29:55 Sola 修改输出网格为ny, nx形式
            2022-08-21 12:25:09 Sola 增加对非经纬度左下角的支持
            2022-08-21 16:27:56 Sola 修正返回网格id数组类型为float的问题
            2022-09-27 22:19:15 Sola 简化网格生成方法
            2022-09-28 16:41:03 Sola v2 加入了列表识别, 根据__iter__属性识别合适方法
            2022-09-28 16:41:38 Sola v2 加入了检测proj是否包含坐标转换的方法
            2022-09-28 16:42:12 Sola v2 加入了转化传入对象为numpy数组的功能
            2022-09-28 18:28:38 Sola v2 修正了计算网格id时, 未输出ix, iy的bug
            2023-03-14 10:02:41 Sola v3 增加输出边界网格的功能(调整get_grid, 使其支持边界宽度及边缘网格id)
            2023-03-18 15:17:40 Sola v4 删除扩展边界的选项
            2023-03-18 15:18:04 Sola v4 修正输入高维数组时, 计算报错的问题
            2023-03-18 16:22:17 Sola v5 增加支持获取加密网格的方法, 用于超采样清单
            2023-03-19 21:53:51 Sola v0.0.2 加入了默认的网格(经纬度网格), 以方便了解功能
            2023-04-29 18:54:06 Sola v0.0.3 加入了从WRF读取数据, 以及输出cartopy.crs的功能
            2023-09-07 10:42:59 Sola v0.0.4 设定了默认的地球形状, 以修正默认投影与模式的偏差, 加入globe参数
                感谢韩雨阳的帮助, 指出了两个差异的问题所在
            2023-12-28 15:42:11 Sola v0.0.5 增加了加密网格的功能
            2023-12-28 15:54:53 Sola v0.0.6 增加了获取绘图范围的功能, 并使其接受浮点数输入
            2024-07-22 20:36:52 Sola v0.0.7 增加了判断坐标（坐标数组）是否在模式网格内的功能
            2024-12-18 10:11:55 Sola v0.0.8 增加了与墨卡托投影相关的计算内容
            2025-04-06 16:39:26 Sola v0.0.9 增加提供网格中心坐标计算网格的功能（优先级低于左下角坐标）
            2025-04-06 16:45:22 Sola v0.0.10 增加坐标旋转功能
                修改的关键在于：
                1. 在将经纬度转化为网格的时候, 围绕中心对网格进行偏移旋转, 需要增加一步后处理
                2. 在将网格转化为经纬度的时候, 需要先将输入的网格ID旋转回去, 再计算其经纬度
                设计的网格旋转函数需要保证旋转前后中心位置不变，各网格相对位置不变即可
                注意, 这里输入的左下角坐标与通过中心计算的左下角坐标均为旋转前的
            2025-07-14 15:42:22 Sola v0.0.11 增加select方法, 用于选取某个经纬度范围的数据
        测试记录:
            2022-09-28 16:28:10 Sola v2 新的简化网格生成方法测试完成, 结果与旧版一致
            2022-09-28 18:27:59 Sola v2 测试了使用proj_LC投影的相关方法, 网格与WRF一致
        """
        try:
            self.type = 'lonlat' if type is None else type # 类型
            self.nx = 360 if nx is None else nx # x方向网格数
            self.ny = 180 if ny is None else ny # y 方向网格数
            self.projection = ccrs.PlateCarree() if proj is None else proj # 投影类别, 使用cartopy的crs提供
            self.dx = 1 if dx is None else dx # 在该投影下x方向间距
            self.dy = 1 if dy is None else dy # 在该投影下y方向间距
            self.dt = 1 if dt is None else dt # 时间间隔(小时)
            self.nt = 1 if nt is None else nt # 每个文件中包含多少时间点
            self.var_list = [] if var_list is None else var_list # 变量列表
            self.globe = ccrs.Globe(ellipse="sphere", semimajor_axis=6370000, semiminor_axis=6370000) if globe is None else globe
            if lowerleft is None:
                if not center is None:
                    center_x, center_y = self.projection.transform_point(center[0], center[1], ccrs.PlateCarree())
                else:
                    center_x, center_y = 0, 0
                zero_lon, zero_lat = ccrs.PlateCarree().transform_point(
                    center_x-self.dx*(self.nx-1)/2, center_y-self.dy*(self.ny-1)/2, self.projection)
                self.lowerleft = [zero_lon, zero_lat] # 旋转前的左下角坐标
            else:
                if len(lowerleft) == 2:
                    self.lowerleft = lowerleft # 旋转前的左下角坐标(经纬度)
                else:
                    # 这是考虑输入的左下角坐标不是经纬度, 而是某个投影系下的坐标位置, 所以先将其转化为经纬度
                    zero_lon, zero_lat = ccrs.PlateCarree().transform_point(\
                        lowerleft[0], lowerleft[1], lowerleft[2])
                    self.lowerleft = [zero_lon, zero_lat] # 旋转前的左下角经纬度
            self.lowerleft_projxy = self.projection.transform_point(
                self.lowerleft[0], self.lowerleft[1],
                ccrs.PlateCarree()
            ) # 计算投影下的xy坐标
            self.rotate = 0 if rotate_deg is None else np.deg2rad(rotate_deg) # 计算旋转的弧度(输入是角度)
            if rotate_poi is None:
                # 如果没有给定围绕旋转的点位, 则围绕网格中心进行旋转, 注意这里是 (x, y), 而不是 (ix, iy)
                # 注意需要考虑如果指定的网格中心和投影中心不一致的情况
                if not center is None:
                    self.rotate_poi_x, self.rotate_poi_y = center_x, center_y
                else:
                    self.rotate_poi_x, self.rotate_poi_y = self.lowerleft_projxy[0] + (self.nx - 1)*self.dx, self.lowerleft_projxy[1] + (self.ny - 1)*self.dy
            else:
                # 如果
                self.rotate_poi_x, self.rotate_poi_y = self.projection.transform_point(*rotate_poi, ccrs.PlateCarree())
        finally:
            if debug > 0:
                print(f"{self.__dict__}")

    def grid_id_float(self, original_x, original_y, original_proj=ccrs.PlateCarree()):
        """
        获取经纬度对应的网格xy值, 返回浮点数
        2022-09-28 11:05:09 Sola 更新为识别传入的对象类型, 判断是否可迭代
        2022-09-28 15:21:07 Sola 增加对proj是否包含相应方法的识别
        2022-09-28 18:25:24 Sola 修正正常情况下未输出ix, iy的bug
        2025-04-06 20:33:29 Sola 加入坐标旋转的判断
        """
        # 如果是可迭代对象, 则丢给对应的功能处理
        if hasattr(original_x, '__iter__'):
            ix, iy = self.grid_ids_float(original_x, original_y, original_proj)
        else: # 如果非可迭代对象, 就有函数本体进行计算
            # 判断投影本身是否具有计算网格ID方法
            if hasattr(self.projection, 'grid_id_float'):
                if original_proj != ccrs.PlateCarree():
                    # 如果有, 且传入坐标非经纬度坐标, 就将其转化为经纬度坐标
                    lon, lat = ccrs.PlateCarree().transform_point(
                        original_x, original_y, original_proj)
                else: # 否则直接使用xy作为经纬度坐标
                    lon, lat = original_x, original_y
                # 调用proj的方法计算经纬度
                ix, iy = self.projection.grid_id_float(lon, lat)
            else: # 如果投影方法本身不具备计算网格ID的方法, 那就手动计算网格
                x, y = self.projection.transform_point(original_x, original_y, original_proj)
                ix = (x - self.lowerleft_projxy[0])/self.dx
                iy = (y - self.lowerleft_projxy[1])/self.dy
            ix, iy = self.rotate_grid_revise(ix, iy)
        return ix, iy

    def grid_id(self, original_x, original_y, original_proj=ccrs.PlateCarree()):
        """
        获取经纬度最近的网格xy值, 返回整数
        2022-09-28 15:29:32 Sola 增加判断传入的是单个值还是可迭代数组的功能
        """
        if hasattr(original_x, '__iter__'): # 如果传入的是可迭代对象, 则丢给对应函数处理
            ix, iy = self.grid_ids(original_x, original_y, original_proj)
        else: # 如果传入的是单个数值, 则由对应功能计算浮点数坐标, 然后取整
            ix, iy = self.grid_id_float(original_x, original_y, original_proj)
            ix, iy = [round(n) for n in [ix, iy]]
        return ix, iy
    
    def grid_ids_float(self, original_x_array, original_y_array,
        original_proj=ccrs.PlateCarree()):
        """
        将经纬度向量或矩阵转换为网格xy值, 返回浮点数
        2022-08-21 16:34:07 Sola 修正了忘了求网格的错误(这错误太不应该了)
        2022-08-21 17:53:45 Sola 修正了两个ix_array的错误(复制粘贴的恶果)
        2022-09-28 15:34:39 Sola 增加判断proj是否由计算网格id的功能
        2022-09-28 15:46:50 Sola 简化原本的网格计算, 使用转置的方式代替判断返回数组长度
        2022-09-28 16:40:27 Sola 增加将输入数组转化为numpy数组的功能, 防止传入列表
        2022-10-19 18:52:25 Sola 修正了除错距离的bug
        2023-03-18 15:39:06 Sola 在计算前, 先将数组展开到1维, 返回时折叠
        2025-04-06 20:33:12 Sola 加入坐标旋转的判断
        注意事项:
            当前存在一个bug, 输入的投影必须是cartopy的投影, 否则无法计算经纬度,
            但是是否有必要在自己写的proj中加入该功能? 需要考虑
        """
        original_x_array, original_y_array, shape = flat_array(\
            np.array(original_x_array), np.array(original_y_array))
        if hasattr(self.projection, 'grid_ids_float'): # 如果投影有相应方法
            # 判断是否是经纬度坐标, 不是则转化为经纬度坐标
            if original_proj != ccrs.PlateCarree():
                lon_array, lat_array, _ = ccrs.PlateCarree().transform_points(
                    original_proj, original_x_array, original_y_array).T
                lon_array, lat_array = lon_array.T, lat_array.T
            else: # 如果是经纬度坐标, 则使用原来的数据
                lon_array, lat_array = original_x_array, original_y_array
            # 调用投影的坐标计算方法进行计算
            ix_array, iy_array = self.projection.grid_ids_float(
                lon_array, lat_array)
        else: # 如果没有, 则采用默认方法
            ix_array, iy_array, _ = self.projection.transform_points(
                original_proj, original_x_array, original_y_array
            ).T # 计算转换后的坐标(m)(转置后)
            # 将m转化为网格坐标
            ix_array = ((ix_array - self.lowerleft_projxy[0])/ self.dx).T
            iy_array = ((iy_array - self.lowerleft_projxy[1])/ self.dy).T
        ix_array, iy_array = fold_array(ix_array, iy_array, shape)
        ix_array, iy_array = self.rotate_grid_revise(ix_array, iy_array)
        return ix_array, iy_array
    
    def grid_ids(self, original_x_array, original_y_array,
        original_proj=ccrs.PlateCarree()):
        """
        将经纬度向量或矩阵转换为网格xy值, 返回整数
        2022-08-21 16:30:39 Sola 修正了返回数组类型为float的问题
        """
        ix_array, iy_array = self.grid_ids_float(
            original_x_array, original_y_array, original_proj
        )
        ix_array, iy_array = [np.round(n_array) for n_array in [ix_array, iy_array]]
        return ix_array.astype(int), iy_array.astype(int)
    
    def grid_lonlat(self, ix, iy):
        """
        通过网格id获取经纬度坐标
        2022-09-28 16:03:27 Sola 增加判断传入的是数值还是数组的功能
        2022-09-28 16:05:07 Sola 增加判断proj是否有计算网格的功能
        2025-04-06 20:32:55 Sola 加入坐标旋转的判断
        """
        if hasattr(ix, '__iter__'): # 如果传入的是可迭代对象, 则调用相应功能
            lon, lat = self.grid_lonlats(ix, iy)
        else: # 如果不是, 则由本函数继续运算
            ix, iy = self.rotate_grid(ix, iy)
            if hasattr(self.projection, 'grid_lonlat'): # 如果投影本身可以计算
                lon, lat = self.projection.grid_lonlat(ix, iy) # 计算网格对应经纬度
            else: # 如果投影不能根据网格ID计算经纬度, 则手动计算
                # 这里则是根据网格计算了在给定投影下的坐标XY，然后将其转化为经纬度
                x = self.lowerleft_projxy[0] + ix * self.dx
                y = self.lowerleft_projxy[1] + iy * self.dy
                lon, lat = ccrs.PlateCarree().transform_point(x, y, self.projection)
        return lon, lat
    
    def grid_lonlats(self, ix_array, iy_array):
        """
        通过网格id矩阵获得经纬度坐标矩阵
        2022-09-28 16:07:40 Sola 增加判断proj是否有计算网格的功能
        2022-09-28 16:08:38 Sola 简化原本的网格计算, 使用转置的方式代替判断返回数组长度
        2022-09-28 16:40:27 Sola 增加将输入数组转化为numpy数组的功能, 防止传入列表
        2023-03-18 15:39:06 Sola 在计算前, 先将数组展开到1维, 返回时折叠
        2025-04-06 20:33:56 Sola 加入坐标旋转的判断
        """
        ix_array, iy_array, shape = flat_array(np.array(ix_array), np.array(iy_array))
        ix_array, iy_array = self.rotate_grid(ix_array, iy_array)
        if hasattr(self.projection, 'grid_lonlats'):
            lon_array, lat_array = self.projection.grid_lonlats(ix_array, iy_array)
        else:
            x_array = self.lowerleft_projxy[0] + ix_array * self.dx
            y_array = self.lowerleft_projxy[1] + iy_array * self.dy
            lon_array, lat_array, _ = ccrs.PlateCarree().transform_points(
                self.projection, x_array, y_array).T
            lon_array, lat_array = lon_array.T, lat_array.T
        lon_array, lat_array = fold_array(lon_array, lat_array, shape)
        return lon_array, lat_array
    
    def get_grid(self, type=None):
        """
        范围模式所有网格的经纬度坐标
        type: None | "corner" | "edge" | "corner_2d"
        2023-03-14 10:05:43 Sola 更新边界宽度的功能及边缘网格的功能
                                    获取的边缘网格从左下角开始顺时针排序(左优先)
        2023-03-14 10:30:23 Sola 经过测试, 代码可以正常运行
        2023-03-18 15:40:20 Sola 删除边界宽度的功能(没有用了)
        2024-08-02 18:01:48 Sola 添加生成边界经纬度的功能
        """
        # 获取网格信息, 下标从0开始
        ys, xs = np.meshgrid(range(self.ny), range(self.nx), indexing='ij')
        if type is None:
            xlon, xlat = self.grid_lonlats(xs, ys) # 从网格信息获取经纬度信息
        elif type.lower() in ["corner", "c"]: # 四角的网格 (4, ny, nx)
            result = []
            result.append(self.grid_lonlats(xs - 0.5, ys - 0.5))
            result.append(self.grid_lonlats(xs - 0.5, ys + 0.5))
            result.append(self.grid_lonlats(xs + 0.5, ys + 0.5))
            result.append(self.grid_lonlats(xs + 0.5, ys - 0.5))
            xlon = np.array([x[0] for x in result])
            xlat = np.array([x[1] for x in result])
        elif type.lower() in ["edge", "e"]: # 四边中心的网格
            result = []
            result.append(self.grid_lonlats(xs - 0.5, ys))
            result.append(self.grid_lonlats(xs, ys + 0.5))
            result.append(self.grid_lonlats(xs + 0.5, ys))
            result.append(self.grid_lonlats(xs, ys - 0.5))
            xlon = np.array([x[0] for x in result])
            xlat = np.array([x[1] for x in result])
        elif type.lower() in ["corner_2d", "c2d"]: # 四角网络 (ny + 1, nx + 1)
            ys, xs = np.meshgrid(range(self.ny+1), range(self.nx+1), indexing='ij')
            xlon, xlat = self.grid_lonlats(xs-0.5, ys-0.5)
        return xlon, xlat

    def get_density_grid(self, density=10, flat=False):
        """
        获取一个更密的网格, 原先的每个网格均匀返回多个点, 例如返回10*10=100个点
            可用于超采样, 以进行清单的分配等操作, 注意不要设置太大的密度, 否则
            可能内存会寄
        更新记录:
            2023-03-18 16:09:39 Sola 编写源代码
            2023-03-18 16:21:46 Sola 测试功能正常, 从网格到经纬度及反向都正常
            2023-10-18 16:19:10 Sola 增加了将结果展开成2D的功能
            2023-12-28 15:38:53 Sola 调整了数组顺序, 方便最终展开
        """

        jj, sub_jj, ii, sub_ii = np.meshgrid(range(self.ny), range(density), 
            range(self.nx), range(density), indexing='ij')
        fii = ii - 0.5 + (sub_ii + 0.5)/density
        fjj = jj - 0.5 + (sub_jj + 0.5)/density
        if flat:
            fii = fii.reshape((self.ny*density, self.nx*density))
            fjj = fjj.reshape((self.ny*density, self.nx*density))
        xlonf, xlatf = self.grid_lonlats(fii, fjj)

        return xlonf, xlatf
    
    def get_ccrs(self):
        """
        获取用于绘图的地图投影, 目前只支持兰伯特投影
        """
        if type(self.projection) is proj_LC:
            proj = ccrs.LambertConformal(
                central_longitude   = self.projection.stdlon,
                standard_parallels  = [
                    self.projection.truelat1,
                    self.projection.truelat2,
                ],
                globe = self.globe
            )
        elif type(self.projection) is proj_MERC:
            proj = ccrs.Mercator(
                central_longitude=self.projection.stdlon,
                globe = self.globe
            )
        elif self.projection.__class__.__base__ is ccrs.Projection:
            proj = self.projection
        else:
            proj = ccrs.PlateCarree(globe = self.globe)
        return proj
    
    def get_extent(
        self,
        cx      : float = None,
        cy      : float = None,
        dx      : float = None,
        dy      : float = None,
        ratio   : float = 1
    ) -> list:
        """
        用于获取指定数据范围的经纬度坐标
        参数:
            cx: 中心点x坐标
            cy: 中心点y坐标
            dx: 中心点周围x网格数
            dy: 中心点周围y网格数
        """
        if cx is None:
            cx, cy, dx, dy = self.nx/2, self.ny/2, self.nx/2, self.ny/2
        # XLON, XLAT = self.get_grid()
        # ys, ye, xs, xe = np.floor(cy-dy), np.ceil(cy+dy), np.floor(cx-dx), np.ceil(cx+dx)
        lon_start, _ = self.grid_lonlat(cx-dx*ratio, cy)
        lon_end, _ = self.grid_lonlat(cx+dx*ratio, cy)
        _, lat_start = self.grid_lonlat(cx, cy-dy*ratio)
        _, lat_end = self.grid_lonlat(cx, cy+dy*ratio)
        # if lon_start > lon_end:
        #     lon_end += 360
        # XLON, XLAT = XLON[cy-dy:cy+dy, cx-dx:cx+dx], XLAT[cy-dy:cy+dy, cx-dx:cx+dx]
        # clon, clat = np.mean(XLON), np.mean(XLAT)
        # dlon, dlat = (np.max(XLON) - np.min(XLON))/2*ratio, (np.max(XLAT) - np.min(XLAT))/2*ratio
        # clon, clat = (lon_end + lon_start)/2, (lat_end + lat_start)/2
        # dlon, dlat = (lon_end - lon_start)/2*ratio, (lat_end - lat_start)/2*ratio
        # extent = [(clon-dlon+180)%360-180, (clon+dlon+180)%360-180, clat-dlat if clat-dlat>=-90 else -90, clat+dlat if clat+dlat<=90 else 90]
        constrain_lon = lambda x: (x+180)%360-180
        constrain_lat = lambda x: min(abs(x), 90) * (1 if x > 0 else -1)
        extent = [constrain_lon(lon_start), constrain_lon(lon_end), constrain_lat(lat_start), constrain_lat(lat_end)]
        return extent
    
    def is_in_domain(self, origin_x, origin_y, use_float=False):
        """
        用于判断坐标（经纬度）是否在模式网格范围内
        Update:
            2025-05-05 00:13:01 Sola 修正使用浮点数计算时的问题
        """
        if use_float:
            ix, iy = self.grid_id_float(origin_x, origin_y)
        else:
            ix, iy = self.grid_id(origin_x, origin_y)
        result = (0 <= ix) & (ix <= self.nx - 1) & (0 <= iy) & (iy <= self.ny - 1)
        return result

    def rotate_xy(self, x, y, rotate_rad=None):
        rotate_rad = self.rotate if rotate_rad is None else rotate_rad
        x_new, y_new = rotate_xy(x, y, self.rotate_poi_x, self.rotate_poi_y, rotate_rad)
        return x_new, y_new
    
    def rotate_xy_revise(self, x, y, rotate_rad=None):
        rotate_rad = self.rotate if rotate_rad is None else rotate_rad
        x_new, y_new = rotate_xy(x, y, self.rotate_poi_x, self.rotate_poi_y, -rotate_rad)
        return x_new, y_new
    
    def rotate_grid(self, ix, iy, rotate_rad=None):
        rotate_rad = self.rotate if rotate_rad is None else rotate_rad
        if np.sum(np.abs(rotate_rad % (np.pi*2)) > 1e-8):
            x, y = self.lowerleft_projxy[0] + ix*self.dx, self.lowerleft_projxy[1] + iy*self.dy
            x_new, y_new = self.rotate_xy(x, y, rotate_rad)
            ix_new, iy_new = (x_new - self.lowerleft_projxy[0])/self.dx, (y_new - self.lowerleft_projxy[1])/self.dy
        else:
            ix_new, iy_new = ix, iy
        return ix_new, iy_new
    
    def rotate_grid_revise(self, ix, iy, rotate_rad=None):
        rotate_rad = self.rotate if rotate_rad is None else rotate_rad
        if np.sum(np.abs(rotate_rad % (np.pi*2)) > 1e-8):
            x, y = self.lowerleft_projxy[0] + ix*self.dx, self.lowerleft_projxy[1] + iy*self.dy
            x_new, y_new = self.rotate_xy_revise(x, y, rotate_rad)
            ix_new, iy_new = (x_new - self.lowerleft_projxy[0])/self.dx, (y_new - self.lowerleft_projxy[1])/self.dy
        else:
            ix_new, iy_new = ix, iy
        return ix_new, iy_new
    
    def select(self, data, extent: list = [-180, 180, -90, 90]):
        nx, ny = self.nx, self.ny
        lon_s, lon_e, lat_s, lat_e = extent
        lon_list = np.concatenate([
            np.linspace(lon_s, lon_e, nx-1),
            np.linspace(lon_e, lon_e, ny-1),
            np.linspace(lon_e, lon_s, nx-1),
            np.linspace(lon_s, lon_s, ny-1)
        ])
        lat_list = np.concatenate([
            np.linspace(lat_s, lat_s, nx-1),
            np.linspace(lat_s, lat_e, ny-1),
            np.linspace(lat_e, lat_e, nx-1),
            np.linspace(lat_e, lat_s, ny-1)
        ])
        x_list, y_list = self.grid_id_float(lon_list, lat_list)
        limit_range = lambda x, xs, xe: x + (xs - x)*(x < xs) - (x - (xe - 1))*(x > (xe - 1))
        x_s, x_e = int(np.ceil(limit_range(np.min(x_list), -0.5, nx-0.5))), int(np.floor(limit_range(np.max(x_list), -0.5, nx-0.5)) + 1)
        y_s, y_e = int(np.ceil(limit_range(np.min(y_list), -0.5, ny-0.5))), int(np.floor(limit_range(np.max(y_list), -0.5, ny-0.5)) + 1)
        data_select = data[y_s:y_e, x_s:x_e]
        return data_select


def rotate_xy(xx, yy, cx, cy, rad):
    xx_offset = (xx - cx)*np.cos(rad) - (yy - cy)*np.sin(rad)
    yy_offset = (xx - cx)*np.sin(rad) + (yy - cy)*np.cos(rad)
    xx_new, yy_new = cx + xx_offset, cy + yy_offset
    return xx_new, yy_new

def flat_array(
        x : np.ndarray,
        y : np.ndarray
) -> tuple:

    """
    用于将数组展开, 并检查数组性质是否一致
    更新记录:
        2023-03-18 15:25:30 Sola 编写源代码
    """
    
    shape = x.shape
    if not shape == y.shape:
        print(f"[WARNING] dimension mismatch, {x.shape}, {y.shape}")
    x, y = x.reshape(-1), y.reshape(-1)
    
    return x, y, shape


def fold_array(
        x       : np.ndarray,
        y       : np.ndarray,
        shape   : tuple
) -> tuple:

    """
    用于将展开的数组折叠回去
    更新记录:
        2023-03-18 15:26:42 Sola 编写源代码
    """
    
    x, y = x.reshape(shape), y.reshape(shape)
    
    return x, y


def from_wrf(file: str) -> model_info_2d:
    """
    接受一个文件路径, 其应当是一个由WPS或WRF输出的文件, 包含了WRF模式网格的相关
        信息. 仅识别兰伯特投影
    更新记录:
        2023-04-29 18:36:30 Sola 编写源代码
    """
    
    # import need library
    import netCDF4 as nc
    # open dataset
    with nc.Dataset(file) as nf:
        dx, dy = nf.DX, nf.DY
        nx, ny = nf.dimensions["west_east"].size, nf.dimensions["south_north"].size
        truelat1 = nf.TRUELAT1
        stdlon = nf.STAND_LON
        lat1, lon1 = nf.CEN_LAT, nf.CEN_LON
        if nf.MAP_PROJ == 1: # Lambert proj
            truelat2 = nf.TRUELAT2
            # make projection
            proj = proj_LC(dx=dx, dy=dy, truelat1=truelat1, truelat2=truelat2,
                        lat1=lat1, lon1=lon1, stdlon=stdlon, nx=nx, ny=ny)
        elif nf.MAP_PROJ == 3: # Mercator proj
            proj = proj_MERC(dx=dx, dy=dy, truelat1=truelat1, lat1=lat1,
                             lon1=lon1, stdlon=stdlon, nx=nx, ny=ny)
        elif nf.MAP_PROJ == 6: # lon-lat proj
            proj = ccrs.PlateCarree
            dx /= 111177.473
            dy /= 111177.473
            
        # make model_info
        model = model_info_2d(proj=proj, nx=nx, ny=ny, dx=dx, dy=dy,
                            lowerleft=proj.grid_lonlat(0, 0))
    return model

def from_ctl(file: str) -> model_info_2d:
    with open(file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "PDEF" in line.upper():
            if "LCC" in line.upper():
                _, nx, ny, _, lat1, lon1, knowi, knowj, truelat1, truelat2, stdlon, dx, dy = line.split()
                nx, ny = int(nx), int(ny)
                lat1, lon1, knowi, knowj, truelat1, truelat2, stdlon, dx, dy =\
                    float(lat1), float(lon1), float(knowi), float(knowj), float(truelat1), float(truelat2), float(stdlon), float(dx), float(dy)
                proj = proj_LC(dx=dx, dy=dy, truelat1=truelat1, truelat2=truelat2, lat1=lat1, lon1=lon1,
                    knowni=knowi, knownj=knowj, stdlon=stdlon, nx=nx, ny=ny)
    model = model_info_2d(proj=proj, nx=nx, ny=ny, dx=dx, dy=dy, lowerleft=proj.grid_lonlat(0, 0))
    return model
