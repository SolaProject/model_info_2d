import numpy as np
from math import radians, cos, tan, log10, sin, sqrt, atan2, atan, degrees
import logging

"""
更新记录:
    2022-09-23 11:59:01 Sola v1 编写源代码, 修正set_lc代码错误的问题
    2024-12-16 09:54:40 Sola v2 增加变量检测的内容
    2024-12-17 15:39:45 Sola v3 增加墨卡托投影
"""

logging.basicConfig(format='[%(asctime)s][%(levelname)s]: %(message)s',
                    level=logging.DEBUG, datefmt='%Y-%m-%dT%H:%M:%S %Z')
EARTH_RADIUS_M = 6370000.

class proj_info(object):
    """参考WPS的geogrid源代码写的"""
    def __init__(self, code, lat1=None, lon1=None, lat0=None, lon0=None,
        dx=None, dy=None, latinc=None, loninc=None, stdlon=None, 
        truelat1=None, truelat2=None, phi=None, lambda1=None, 
        ixdim=None, jydim=None, stagger='HH', nlat=None, nlon=None, nxmin=None, 
        nxmax=None, hemi=None, cone=None, polei=None, polej=None, 
        rsw=None, knowni=None, knownj=None, re_m=EARTH_RADIUS_M, 
        init=False, wrap=False, rho0=None, nc=None, bigc=None, comp_ll=False, 
        gauss_lat=None, nx=None, ny=None) -> None:
        self.code       = code
        self.lat1       = lat1      # SW latitude (1,1) in degrees (-90->90N) 格点(1, 1)纬度, 西南角, 度
        self.lon1       = lon1      # SW longitude (1,1) in degrees (-180->180E) 格点(1, 1)经度, 西南角, 度
        self.lat0       = lat0
        self.lon0       = lon0
        self.dx         = dx        # Grid spacing in meters at truelats, used x方向网格距, m
        self.dy         = dy        # Grid spacing in meters at truelats, used y方向网格距, m
        self.latinc     = latinc
        self.loninc     = loninc
        self.stdlon     = stdlon    # Longitude parallel to y-axis (-180->180E) 中央经线, 与网格上所有y方向的线平行
        self.truelat1   = truelat1  # First true latitude (all projections) 标准纬线1, 所有投影都需要
        self.truelat2   = truelat2  # Second true lat (LC only) 标准纬线2, 仅兰伯特投影需要
        self.phi        = phi
        self.lambda1    = lambda1
        self.ixdim      = ixdim
        self.jydim      = jydim
        self.stagger    = stagger
        self.nlat       = nlat
        self.nlon       = nlon
        self.nxmin      = nxmin
        self.nxmax      = nxmax
        self.hemi       = hemi      # 1 for NH, -1 for SH 判断位于哪个半球, 北半球则为1, 南半球则为-1
        self.cone       = cone      # Cone factor for LC projections 兰伯特投影的角度因素
        self.polei      = polei     # Computed i-location of pole point 极点的i坐标
        self.polej      = polej     # Computed j-location of pole point 极点的j坐标
        self.rsw        = rsw       # Computed radius to SW corner 西南角(左下角)半径
        self.knowni     = knowni    # X-location of known lat/lon 已知的点位的i坐标
        self.knownj     = knownj    # Y-location of known lat/lon 已知的点位的j坐标（一般就是ctl文件里面的那个点，也就是中心点）
        self.re_m       = re_m      # Radius of spherical earth, meters 地球半径, 6370000, m
        self.init       = init
        self.wrap       = wrap
        self.rho0       = rho0
        self.nc         = nc
        self.bigc       = bigc
        self.comp_ll    = comp_ll
        self.gauss_lat  = gauss_lat
        self.nx         = nx
        self.ny         = ny

        if self.knowni is None and self.knownj is None:
            self.knowni = (self.nx + 1) / 2
            self.knownj = (self.ny + 1) / 2
        if self.lat1:
            if abs(self.lat1) > 90:
                logging.error("Latitude of origin corner required as follows: -90N <= lat1 < = 90.N")
        if self.lon1: # 限制经度范围
            dummy_lon1 = (self.lon1 + 180) % 360 - 180
            self.lon1 = dummy_lon1
        if self.lon0: # 限制中央经线范围
            dummy_lon0 = (self.lon0 + 180) % 360 - 180
            self.lon0 = dummy_lon0
        if self.dx:
            if self.dx <= 0 and self.code != "PROJ_LATLON":
                logging.error("Require grid spacing (dx) in meters be positive!")
        if self.stdlon:
            dummp_stdlon = (self.stdlon + 180) % 360 - 180
            self.stdlon = dummp_stdlon
        if self.truelat1:
            if abs(self.truelat1) > 90:
                logging.error("Set true latitude 1 for all projections!")
        if not self.dy and self.dx: # 设置dy, 如果dy不存在, 则利用dx给定
            self.dy = self.dx
        if self.dx:
            if self.code in ["PROJ_LC", "PROJ_PS", "PROJ_PS_WGS84", "PROJ_A:NERS_NAD83", "PROJ_MERC"]:
                if self.truelat1 < 0: # 所在半球, 1为北半球, -1为南半球
                    self.hemi = -1
                else:
                    self.hemi = 1
                self.rebydx = self.re_m / self.dx # 地球半径除以网格距
    
    def grid_id_float(self, lon, lat):
        """返回以0为开始的下标"""
        i,j = self.llij(lon, lat)
        return i - 1, j - 1

    def grid_ids_float(self, lon_array, lat_array):
        """返回以0为开始的下标数组"""
        i_array, j_array = self.llij(lon_array, lat_array)
        return i_array - 1, j_array - 1

    def grid_lonlat(self, ix, iy):
        """返回对应网格(以0为下标开始)的经纬度"""
        lon, lat = self.ijll(ix + 1, iy + 1)
        return lon, lat

    def grid_lonlats(self, ix_array, iy_array):
        """返回对应网格数组(以0为下标开始)的经纬度数组"""
        lon_array, lat_array = self.ijll(ix_array + 1, iy_array + 1)
        return lon_array, lat_array
    
    def transform_point(self, lon, lat, proj_useless=None):
        """返回对应经纬度坐标的网格坐标(m)"""
        pass

class proj_LC(proj_info):
    """
    参考WPS源码中的proj_LC改写, 因为WRF计算得到的网格与cartopy的不同
    更新记录:
        2022-09-22 22:07:51 Sola 编写源代码
    """
    def __init__(self, code='PROJ_LC', truelat1=None, truelat2=None, lat1=None,
        lon1=None, knowni=None, knownj=None, stdlon=None, dx=None,
        dy=None, nx=None, ny=None, re_m=EARTH_RADIUS_M) -> None:
        """
        初始化
        必要参数:
            code        投影编码
            truelat1    标准纬线1
            truelat2    标准纬线2
            lat1        参考点纬度
            lon1        参考点经度
            stdlon      中央经线
            dx          x方向网格距(m)
            nx          x方向格点数
            ny          y方向格点数
        可选参数:
            knowni      参考点x方向坐标, 默认为网格中心
            knownj      参考点y方向坐标, 默认为网格中心
            dy          y方向网格距(m), 默认与dx一致
            re_m        地球半径, 默认为6370000
        """
        if truelat1 is None or truelat2 is None or lat1 is None or lon1 is None\
            or nx is None or ny is None or stdlon is None or dx is None:
            print('[ERROR] cannot generate proj!')
        if abs(truelat2) > 90:  # 如果标准纬线2超过范围, 则用标准纬线1赋值
            truelat2 = truelat1
        super().__init__(code=code, lat1=lat1, lon1=lon1, dx=dx, dy=dy, 
            stdlon=stdlon, truelat1=truelat1, truelat2=truelat2, 
            knowni=knowni, knownj=knownj, re_m=re_m, nx=nx, ny=ny) # 初始化各变量
        self.set_lc()           # 计算其他变量
        self.check_init()       # 确认是否所有变量都计算完毕

    def set_lc(self):
        """初始化兰伯特投影"""
        # 这些参数都是从proj结构体中获得的, 目的是获得cone, 一个与圆锥角度相关的量, 应该是正弦值
        self.lc_cone()
         # 左下角网格的经度与中央经线的差
        deltalon1 = self.lon1 - self.stdlon
        # 将这个差值限制在 [-180, 180] 之间
        deltalon1 = (deltalon1 + 180) % 360 - 180
        # Convert truelat1 to radian and compute COS for later use, 将第一条标准纬线计算为余弦值, 接下来要用
        ctl1r = cos(radians(self.truelat1))
        # Compute the radius to our known lower-left (SW) corner 计算距离左下角的半径?
        # 变量说明: rebydx: 地球半径/x方向网格距; hemi: 半球;
        self.rsw = self.rebydx * ctl1r / self.cone * (tan(radians(90*self.hemi \
            - self.lat1)/2)/tan(radians(90*self.hemi - self.truelat1)/2))**self.cone
        # Find pole point 找到极点
        arg = self.cone * radians(deltalon1)
        self.polei = self.hemi * self.knowni - self.hemi * self.rsw * sin(arg)
        self.polej = self.hemi * self.knownj + self.rsw * cos(arg)

    def lc_cone(self):
        """计算切面圆锥的角度"""
        # 首先, 看是割线投影还是切线投影
        if abs(self.truelat1 - self.truelat2) > 0.1:
            self.cone = log10(cos(radians(self.truelat1))) - log10(cos(radians(self.truelat2)))
            self.cone = self.cone/(log10(tan(radians(90-abs(self.truelat1))/2))\
                - log10(tan(radians(90-abs(self.truelat2))/2)))
        else:
            self.cone = sin(radians(abs(self.truelat1)))

    def llij(self, lon, lat):
        """通过经纬度序列计算坐标"""
        if not self.init:
            print('[ERROR] proj cannot use!')
        deltalon = lon - self.stdlon            # 计算经度与中央经线差
        deltalon = (deltalon + 180) % 360 - 180 # 限定范围 -180~180
        ctl1r = np.cos(np.radians(self.truelat1))     # 剩下就看不懂了
        rm = self.rebydx * ctl1r / self.cone * (np.tan(np.radians(90*self.hemi \
            - lat)/2)/np.tan(np.radians(90*self.hemi - self.truelat1)/2))**self.cone
        arg = self.cone * np.radians(deltalon)
        i = self.polei + self.hemi * rm * np.sin(arg)
        j = self.polej - rm * np.cos(arg)
        i = self.hemi * i
        j = self.hemi * j
        return i, j

    def ijll(self, i, j):
        """
        通过坐标计算经纬度
        2022-09-28 18:22:26 Sola 修正计算chi时, xx, yy未进行筛选的问题
        """
        if not self.init:
            print('[ERROR] proj cannot use!')
        chi1 = np.radians(90 - self.hemi * self.truelat1)
        chi2 = np.radians(90 - self.hemi * self.truelat2)
        inew = self.hemi * i
        jnew = self.hemi * j
        xx = np.array(inew - self.polei)
        yy = np.array(self.polej - jnew)
        r2 = xx**2 + yy**2
        r = np.sqrt(r2) / self.rebydx
        lon = np.zeros(r2.shape)
        lat = np.zeros(r2.shape)
        select_array = r2==0
        lat[select_array] = self.hemi * 90
        lon[select_array] = self.stdlon
        lon[~select_array] = self.stdlon + np.degrees(np.arctan2(
            self.hemi*xx[~select_array], yy[~select_array]))/self.cone
        lon[~select_array] = (lon[~select_array] + 180) % 360 - 180
        chi = np.zeros(r2.shape)
        chi[chi1==chi2] = 2 * np.arctan((r/np.tan(chi1))**(1/self.cone) * np.tan(chi1*0.5))
        chi[chi1!=chi2] = 2 * np.arctan((r*self.cone/np.sin(chi1))**(1/self.cone) * np.tan(chi1*0.5))
        lat[~select_array] = (90 - np.degrees(chi[~select_array])) * self.hemi
        return lon, lat

    def check_init(self):
        """确认是否所有变量都已经准备好了"""
        if self.cone is None or self.rsw is None or self.polei is None or \
            self.polej is None:
            print('[ERROR] cannot set proj_lc!')
        else:
            self.init = True

    def transform_point(self, lon, lat, proj_useless=None):
        """返回对应经纬度坐标的网格坐标(m)"""
        ix, iy = self.llij(lon, lat)
        cx, cy1 = self.llij(self.stdlon, self.truelat1)
        cx, cy2 = self.llij(self.stdlon, self.truelat2)
        cy = (cy1 + cy2) / 2
        return (ix - cx) * self.dx, (iy - cy) * self.dy


class proj_MERC(proj_info):
    """
    参考WPS源码中的proj_MERC改写, 因为WRF计算得到的网格与cartopy的不同
    更新记录:
        2024-12-17 15:12:51 Sola 编写源代码
    """
    def __init__(self, code='PROJ_MERC', truelat1=None, lat1=None,
        lon1=None, knowni=None, knownj=None, stdlon=None, dx=None,
        dy=None, nx=None, ny=None, re_m=EARTH_RADIUS_M) -> None:
        """
        初始化
        必要参数:
            code        投影编码
            truelat1    标准纬线1
            lat1        参考点纬度
            lon1        参考点经度
            stdlon      中央经线
            dx          x方向网格距(m)
            nx          x方向格点数
            ny          y方向格点数
        可选参数:
            knowni      参考点x方向坐标, 默认为网格中心
            knownj      参考点y方向坐标, 默认为网格中心
            dy          y方向网格距(m), 默认与dx一致
            re_m        地球半径, 默认为6370000
        """
        if truelat1 is None is None or lat1 is None or lon1 is None\
            or nx is None or ny is None or dx is None:
            print('[ERROR] cannot generate proj!')
        super().__init__(code=code, lat1=lat1, lon1=lon1, dx=dx, dy=dy, nx=nx, ny=ny,
            stdlon=stdlon, truelat1=truelat1, knowni=knowni, knownj=knownj, re_m=re_m) # 初始化各变量
        self.set_merc()           # 计算其他变量

    def set_merc(self):
        clain = np.cos(np.deg2rad(self.truelat1)) # 标准纬线在赤道面投影到地心的距离/地球半径
        self.dlon = self.dx / (self.re_m * clain) # 标准纬线附近，单位网格的经度变化
        # 计算原点到赤道的距离，并保存在 self.rsw 变量中
        self.rsw = 0 if self.lat1 == 0 else np.log(np.tan(0.5*(np.deg2rad(self.lat1+90))))/self.dlon

    def llij(self, lon, lat):
        deltalon = lon - self.lon1
        deltalon = (deltalon + 180) % 360 - 180
        i = self.knowni + deltalon / np.rad2deg(self.dlon)
        j = self.knownj + np.log(np.tan(0.5*np.deg2rad(lat + 90)))/self.dlon - self.rsw
        return i, j
    
    def ijll(self, i, j):
        lat = 2*np.rad2deg(np.arctan(np.exp(self.dlon*(self.rsw + j - self.knownj)))) - 90
        lon = (i - self.knowni) * np.rad2deg(self.dlon) + self.lon1
        lon = (lon + 180) % 360 - 180
        return lon, lat

if __name__ == '__main__':
    """
    主要用于测试投影的基本功能，坐标前后转换是否正常，具体测试项如下：
    1. 构建投影
    2. 坐标点位转换以及转换结果是否一致
    3. 坐标序列转换以及转换结果是否一致
    """
    logging.info("test proj_LC")
    try:
        proj = proj_LC(truelat1=45, truelat2=15, lat1=30, lon1=108, stdlon=108, dx=3000, dy=3000, nx=2025, ny=2025)
        x0, y0 = 0, 0
        lon, lat = proj.grid_lonlat(x0, y0)
        x1, y1 = proj.grid_id_float(lon, lat)
        if not ((x0 - x1)**2 + (y0 - y1)**2)**0.5 < 1e-5:
            logging.error((x0, y0), (x1, y1))
            raise ValueError("point convert error!")
        x0_array, y0_array = np.arange(1, 2000, 10), np.arange(1, 2000, 10)
        lon_array, lat_array = proj.grid_lonlat(x0_array, y0_array)
        x1_array, y1_array = proj.grid_id_float(lon_array, lat_array)
        if not np.max(((x0_array - x1_array)**2 + (y0_array - y1_array)**2)**0.5) < 1e-5:
            raise ValueError("array convert error!")
    except Exception as e:
        logging.error("proj_LC not pass!")
        logging.error(e)
    else:
        logging.info("proj_LC pass.")
    
    logging.info("test proj_MERC")
    try:
        proj = proj_MERC(truelat1=30, lat1=30, lon1=108, stdlon=108, dx=3000, dy=3000, nx=2025, ny=2025)
        x0, y0 = 0, 0
        lon, lat = proj.grid_lonlat(x0, y0)
        x1, y1 = proj.grid_id_float(lon, lat)
        if not ((x0 - x1)**2 + (y0 - y1)**2)**0.5 < 1e-5:
            logging.error(f"{(x0, y0)}, {(x1, y1)}")
            raise ValueError("point convert error!")
        x0_array, y0_array = np.arange(1, 2000, 10), np.arange(1, 2000, 10)
        lon_array, lat_array = proj.grid_lonlat(x0_array, y0_array)
        x1_array, y1_array = proj.grid_id_float(lon_array, lat_array)
        if not np.max(((x0_array - x1_array)**2 + (y0_array - y1_array)**2)**0.5) < 1e-5:
            raise ValueError("array convert error!")
    except Exception as e:
        logging.error("proj_MERC not pass!")
        logging.error(e)
    else:
        logging.info("proj_MERC pass.")