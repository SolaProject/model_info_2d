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
        """
        x, y = self.projection.transform_point(original_x, original_y, original_proj)
        ix = (x - self.lowerleft_projxy[0])/self.dx
        iy = (y - self.lowerleft_projxy[1])/self.dy
        return ix, iy

    def grid_id(self, original_x, original_y, original_proj=ccrs.PlateCarree()):
        """
        获取经纬度最近的网格xy值, 返回整数
        """
        ix, iy = self.grid_id_float(original_x, original_y, original_proj)
        ix, iy = [round(n) for n in [ix, iy]]
        return ix, iy
    
    def grid_ids_float(self, original_x_array, original_y_array, original_proj=ccrs.PlateCarree()):
        """
        将经纬度向量或矩阵转换为网格xy值, 返回浮点数
        2022-08-21 16:34:07 Sola 修正了忘了求网格的错误(这错误太不应该了)
        2022-08-21 17:53:45 Sola 修正了两个ix_array的错误(复制粘贴的恶果)
        """
        result_array = self.projection.transform_points(
            original_proj, original_x_array, original_y_array
        )
        if len(result_array.shape) == 2:
            result_array[:, 0] = (result_array[:, 0] - self.lowerleft_projxy[0])/self.dx
            result_array[:, 1] = (result_array[:, 1] - self.lowerleft_projxy[1])/self.dy
            ix_array, iy_array = result_array[:, 0], result_array[:, 1]
        else:
            result_array[:, :, 0] = (result_array[:, :, 0] - self.lowerleft_projxy[0])/self.dx
            result_array[:, :, 1] = (result_array[:, :, 1] - self.lowerleft_projxy[1])/self.dy
            ix_array, iy_array = result_array[:, :, 0], result_array[:, :, 1]
        return ix_array, iy_array
    
    def grid_ids(self, original_x_array, original_y_array, original_proj=ccrs.PlateCarree()):
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
        """
        x = self.lowerleft_projxy[0] + ix * self.dx
        y = self.lowerleft_projxy[1] + iy * self.dy
        lon, lat = ccrs.PlateCarree().transform_point(x, y, self.projection)
        return lon, lat
    
    def grid_lonlats(self, ix_array, iy_array):
        """
        通过网格id矩阵获得经纬度坐标矩阵
        """
        x_array = self.lowerleft_projxy[0] + ix_array * self.dx
        y_array = self.lowerleft_projxy[1] + iy_array * self.dy
        result_array = ccrs.PlateCarree().transform_points(self.projection, x_array, y_array)
        if len(result_array.shape) == 2:
            lon_array, lat_array = result_array[:, 0], result_array[:, 1]
        else:
            lon_array, lat_array = result_array[:, :, 0], result_array[:, :, 1]
        return lon_array, lat_array
    
    def get_grid(self):
        """
        范围模式所有网格的经纬度坐标
        """
        zero_x, zero_y = self.lowerleft_projxy
        array_temp = np.arange(zero_x, zero_x+self.nx*self.dx, self.dx)
        xlon = np.reshape(array_temp.repeat(self.ny).T, [self.nx, self.ny]).T
        array_temp = np.arange(zero_y, zero_y+self.ny*self.dy, self.dy)
        xlat = np.reshape(array_temp.repeat(self.nx), [self.ny, self.nx])
        array_temp = ccrs.PlateCarree().transform_points(self.projection, xlon, xlat)
        xlon = array_temp[:, :, 0]
        xlat = array_temp[:, :, 1]
        return xlon, xlat