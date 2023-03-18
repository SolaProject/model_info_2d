import numpy as np
import cartopy.crs as ccrs
class model_info_2d(object):
    """
    用于创建模式网格, 并包含了相关信息, 提供了方便坐标与经纬度相互转换的工具
    基于Numpy和Cartopy.crs构建, 仅支持方形网格
    """
    def __init__(self, proj, nx, ny, dx, dy, lowerleft=None, \
        nt=None, dt=None, var_list=None, type=None) -> None:
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
        测试记录:
            2022-09-28 16:28:10 Sola v2 新的简化网格生成方法测试完成, 结果与旧版一致
            2022-09-28 18:27:59 Sola v2 测试了使用proj_LC投影的相关方法, 网格与WRF一致
        """
        if type is None:
            self.type = 'unknown'
        else:
            self.type = type # 类型
        self.nx = nx # x方向网格数
        self.ny = ny # y方向网格数
        self.projection = proj # 投影类别, 使用cartopy的crs提供
        self.dx = dx # 在该投影下x方向间距
        self.dy = dy # 在该投影下y方向间距
        if dt is None: # 时间间隔(小时)
            self.dt = 1
        else:
            self.dt = dt
        if nt is None: # 每个文件中包含多少时间点
            self.nt = 1
        else:
            self.nt = nt
        if lowerleft is None:
            zero_lon, zero_lat = ccrs.PlateCarree().transform_point(\
                -dx*(nx-1)/2, -dy*(ny-1)/2, proj)
            self.lowerleft = [zero_lon, zero_lat]
        else:
            if len(lowerleft) == 2:
                self.lowerleft = lowerleft # 左下角坐标(经纬度)
            else:
                zero_lon, zero_lat = ccrs.PlateCarree().transform_point(\
                    lowerleft[0], lowerleft[1], lowerleft[2])
                self.lowerleft = [zero_lon, zero_lat]
        if var_list is None: # 变量列表
            self.var_list = []
        else:
            self.var_list = var_list
        self.lowerleft_projxy = self.projection.transform_point(
            self.lowerleft[0], self.lowerleft[1],
            ccrs.PlateCarree()
        ) # 计算投影下的坐标

    def grid_id_float(self, original_x, original_y, original_proj=ccrs.PlateCarree()):
        """
        获取经纬度对应的网格xy值, 返回浮点数
        2022-09-28 11:05:09 Sola 更新为识别传入的对象类型, 判断是否可迭代
        2022-09-28 15:21:07 Sola 增加对proj是否包含相应方法的识别
        2022-09-28 18:25:24 Sola 修正正常情况下未输出ix, iy的bug
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
                x, y = self.projection.transform_point(
                    original_x, original_y, original_proj)
                ix = (x - self.lowerleft_projxy[0])/self.dx
                iy = (y - self.lowerleft_projxy[1])/self.dy
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
        注意事项:
            当前存在一个bug, 输入的投影必须是cartopy的投影, 否则无法计算经纬度,
            但是是否有必要在自己写的proj中加入该功能? 需要考虑
        """
        original_x_array = np.array(original_x_array)
        original_y_array = np.array(original_y_array)
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
        """
        if hasattr(ix, '__iter__'): # 如果传入的是可迭代对象, 则调用相应功能
            lon, lat = self.grid_lonlats(ix, iy)
        else: # 如果不是, 则由本函数继续运算
            if hasattr(self.projection, 'grid_lonlat'): # 如果投影本身可以计算
                lon, lat = self.projection.grid_lonlat(ix, iy) # 计算网格对应经纬度
            else: # 如果投影不能根据网格ID计算经纬度, 则手动计算
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
        """
        ix_array, iy_array = np.array(ix_array), np.array(iy_array)
        if hasattr(self.projection, 'grid_lonlats'):
            lon_array, lat_array = self.projection.grid_lonlats(ix_array, iy_array)
        else:
            x_array = self.lowerleft_projxy[0] + ix_array * self.dx
            y_array = self.lowerleft_projxy[1] + iy_array * self.dy
            lon_array, lat_array, _ = ccrs.PlateCarree().transform_points(
                self.projection, x_array, y_array).T
            lon_array, lat_array = lon_array.T, lat_array.T
        return lon_array, lat_array
    
    def get_grid(self, bdy_width=0, type=None):
        """
        范围模式所有网格的经纬度坐标
        2023-03-14 10:05:43 Sola 更新边界宽度的功能及边缘网格的功能
                                    获取的边缘网格从左下角开始顺时针排序(左优先)
        2023-03-14 10:30:23 Sola 经过测试, 代码可以正常运行
        """
        # 获取网格信息, 下标从0开始
        ys, xs = np.meshgrid(range(-bdy_width, self.ny + bdy_width),
            range(-bdy_width, self.nx + bdy_width), indexing='ij')
        if type is None:
            xlon, xlat = self.grid_lonlats(xs, ys) # 从网格信息获取经纬度信息
        elif type.lower() in ["corner", "c"]: # 四角的网格
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
        return xlon, xlat