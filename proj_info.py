import numpy as np
from math import radians, cos, tan, log10, sin, sqrt, atan2, atan, degrees

"""
更新记录:
    2022-09-23 11:59:01 Sola v1 编写源代码, 修正set_lc代码错误的问题
"""

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
        gauss_lat=None) -> None:
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
        if abs(lat1) > 90 or dx <= 0 or truelat1 > 90:
            pass
        dummy_lon1 = (lon1 + 180) % 360 - 180       # 限制经度范围
        dummy_stdlon = (stdlon + 180) % 360 - 180   # 限制中央经线范围
        if knowni is None and knownj is None:
            knowni = (nx + 1) / 2
            knownj = (ny + 1) / 2
        if dy is None:          # 设置dy, 如果dy不存在, 则利用dx给定
            dy = dx
        if truelat1 < 0:        # 所在半球, 1为北半球, -1为南半球
            hemi = -1
        else:
            hemi = 1
        if abs(truelat2) > 90:  # 如果标准纬线2超过范围, 则用标准纬线1赋值
            truelat2 = truelat1
        super().__init__(code=code, lat1=lat1, lon1=dummy_lon1, dx=dx, dy=dy, 
            stdlon=dummy_stdlon, truelat1=truelat1, truelat2=truelat2, hemi=hemi, 
            knowni=knowni, knownj=knownj, re_m=re_m) # 初始化各变量
        self.rebydx = re_m / dx # 地球半径除以网格距
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

    def llij_lc(self, lon, lat):
        """通过经纬度计算坐标"""
        if not self.init:
            print('[ERROR] proj cannot use!')
        deltalon = lon - self.stdlon            # 计算经度与中央经线差
        deltalon = (deltalon + 180) % 360 - 180 # 限定范围 -180~180
        ctl1r = cos(radians(self.truelat1))     # 剩下就看不懂了
        rm = self.rebydx * ctl1r / self.cone * (tan(radians(90*self.hemi \
            - lat)/2)/tan(radians(90*self.hemi - self.truelat1)/2))**self.cone
        arg = self.cone * radians(deltalon)
        i = self.polei + self.hemi * rm * sin(arg)
        j = self.polej - rm * cos(arg)
        i = self.hemi * i
        j = self.hemi * j
        return i, j

    def llijs_lc(self, lon, lat):
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

    def ijll_lc(self, i, j):
        """通过坐标计算经纬度"""
        if not self.init:
            print('[ERROR] proj cannot use!')
        chi1 = radians(90 - self.hemi * self.truelat1)
        chi2 = radians(90 - self.hemi * self.truelat2)
        inew = self.hemi * i
        jnew = self.hemi * j
        xx = inew - self.polei
        yy = self.polej - jnew
        r2 = xx**2 + yy**2
        r = sqrt(r2) / self.rebydx
        if r2 == 0.:
            lat = self.hemi * 90
            lon = self.stdlon
        else:
            lon = self.stdlon + degrees(atan2(self.hemi*xx, yy))/self.cone
            lon = (lon + 180) % 360 - 180
            if chi1 == chi2:
                chi = 2 * atan((r/tan(chi1))**(1/self.cone) * tan(chi1*0.5))
            else:
                chi = 2 * atan((r*self.cone/sin(chi1))**(1/self.cone) * tan(chi1*0.5))
            lat = (90 - degrees(chi)) * self.hemi
        return lon, lat

    def ijlls_lc(self, i, j):
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
        xx = inew - self.polei
        yy = self.polej - jnew
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

    def grid_id_float(self, lon, lat):
        """返回以0为开始的下标"""
        i,j = self.llij_lc(lon, lat)
        return i - 1, j - 1

    def grid_ids_float(self, lon_array, lat_array):
        """返回以0为开始的下标数组"""
        i_array, j_array = self.llijs_lc(lon_array, lat_array)
        return i_array - 1, j_array - 1

    def grid_lonlat(self, ix, iy):
        """返回对应网格(以0为下标开始)的经纬度"""
        lon, lat = self.ijll_lc(ix + 1, iy + 1)
        return lon, lat

    def grid_lonlats(self, ix_array, iy_array):
        """返回对应网格数组(以0为下标开始)的经纬度数组"""
        lon_array, lat_array = self.ijlls_lc(ix_array + 1, iy_array + 1)
        return lon_array, lat_array

    def transform_point(self, lon, lat, proj_useless=None):
        """返回对应经纬度坐标的网格坐标(m)"""
        ix, iy = self.llij_lc(lon, lat)
        cx, cy1 = self.llij_lc(self.stdlon, self.truelat1)
        cx, cy2 = self.llij_lc(self.stdlon, self.truelat2)
        cy = (cy1 + cy2) / 2
        return (ix - cx) * self.dx, (iy - cy) * self.dy


if __name__ == '__main__':
    proj = proj_LC(truelat1=45, truelat2=15, lat1=30, lon1=108, stdlon=108, dx=3000, dy=3000, nx=2025, ny=2025)
    proj.llij_lc(108, 30)
    proj.ijll_lc(1013, 1013)
    x = np.arange(1, 2000, 10)
    y = np.arange(1, 2000, 10)
    print(proj.ijlls_lc(x, y))
    print(proj.ijll_lc(x[0], y[0]), proj.ijll_lc(x[-1], y[-1]))
    x, y = proj.llij_lc(0, 90)
    lon, lat = proj.ijll_lc(x, y)
    x, y = proj.llij_lc(lon, lat)
    lon1, lat1 = proj.ijll_lc(x, y)
    print(lon, lat, x, y, lon1, lat1)