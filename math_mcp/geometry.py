# -*- coding: utf-8 -*-
"""
几何计算模块
提供平面几何和立体几何计算功能
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class GeometryCalculator:
    """几何计算器类"""

    def __init__(self):
        self.pi = math.pi

    def geometry_calculator_tool(
        self,
        shape_type: str,
        operation: str,
        dimensions: Optional[Dict[str, float]] = None,
        points: Optional[List[List[float]]] = None,
        precision: Optional[int] = None,
        unit: str = "default",
    ) -> Dict[str, Any]:
        """
        几何计算工具

        Args:
            shape_type: 形状类型
            operation: 操作类型
            dimensions: 尺寸参数字典
            points: 坐标点列表
            precision: 计算精度（小数位数）
            unit: 计量单位
        """
        # 参数验证
        validation_result = self._validate_parameters(
            shape_type, operation, dimensions, points, precision
        )
        if validation_result is not None:
            return validation_result

        try:
            if shape_type in [
                "circle",
                "triangle",
                "rectangle",
                "polygon",
                "ellipse",
                "parallelogram",
                "trapezoid",
                "rhombus",
                "regular_polygon",
            ]:
                return self._plane_geometry(
                    shape_type, operation, dimensions, points, precision, unit
                )
            elif shape_type in [
                "sphere",
                "cube",
                "cylinder",
                "cone",
                "pyramid",
                "prism",
                "torus",
                "ellipsoid",
            ]:
                return self._solid_geometry(
                    shape_type, operation, dimensions, precision, unit
                )
            elif shape_type == "analytical":
                return self._analytical_geometry(operation, points, precision, unit)
            else:
                return {"error": f"不支持的形状类型: {shape_type}"}

        except Exception as e:
            return {"error": f"几何计算错误: {str(e)}"}

    def _validate_parameters(
        self,
        shape_type: str,
        operation: str,
        dimensions: Optional[Dict[str, float]],
        points: Optional[List[List[float]]],
        precision: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """参数验证"""
        # 验证shape_type
        valid_shapes = [
            "circle",
            "triangle",
            "rectangle",
            "polygon",
            "ellipse",
            "parallelogram",
            "trapezoid",
            "rhombus",
            "regular_polygon",
            "sphere",
            "cube",
            "cylinder",
            "cone",
            "pyramid",
            "prism",
            "torus",
            "ellipsoid",
            "analytical",
        ]
        if shape_type not in valid_shapes:
            return {
                "error": f"无效的形状类型: {shape_type}，支持的类型: {valid_shapes}"
            }

        # 验证operation
        valid_operations = [
            "area",
            "volume",
            "surface_area",
            "circumference",
            "perimeter",
            "properties",
            "diagonal",
            "distance",
            "midpoint",
            "slope",
            "line_equation",
            "angle",
            "centroid",
            "incircle",
            "circumcircle",
        ]
        if operation not in valid_operations:
            return {
                "error": f"无效的操作类型: {operation}，支持的操作: {valid_operations}"
            }

        # 验证precision
        if precision is not None and (
            not isinstance(precision, int) or precision < 0 or precision > 15
        ):
            return {"error": "精度必须是0-15之间的整数"}

        # 验证dimensions中的数值
        if dimensions:
            for key, value in dimensions.items():
                if not isinstance(value, (int, float)):
                    return {"error": f"尺寸参数 {key} 必须是数值类型"}
                if (
                    key
                    in [
                        "radius",
                        "length",
                        "width",
                        "height",
                        "side",
                        "base_area",
                        "major_axis",
                        "minor_axis",
                    ]
                    and value <= 0
                ):
                    return {"error": f"尺寸参数 {key} 必须为正数"}

        # 验证points
        if points:
            if not isinstance(points, list):
                return {"error": "坐标点必须是列表格式"}
            for i, point in enumerate(points):
                if not isinstance(point, list):
                    return {"error": f"第{i+1}个点必须是列表格式"}
                if len(point) not in [2, 3]:
                    return {"error": f"第{i+1}个点必须是2D或3D坐标"}
                for j, coord in enumerate(point):
                    if not isinstance(coord, (int, float)):
                        return {"error": f"第{i+1}个点的第{j+1}个坐标必须是数值"}

        return None

    def _plane_geometry(
        self,
        shape_type: str,
        operation: str,
        dimensions: Optional[Dict[str, float]],
        points: Optional[List[List[float]]] = None,
        precision: Optional[int] = None,
        unit: str = "default",
    ) -> Dict[str, Any]:
        """平面几何计算"""
        result = {}

        if shape_type == "circle":
            if dimensions is None or "radius" not in dimensions:
                return {"error": "圆形计算需要提供radius参数"}
            r = dimensions["radius"]

            if operation == "area":
                result["area"] = self.pi * r**2
            elif operation == "circumference":
                result["circumference"] = 2 * self.pi * r
            elif operation == "properties":
                result["area"] = self.pi * r**2
                result["circumference"] = 2 * self.pi * r
                result["diameter"] = 2 * r
                result["radius"] = r

        elif shape_type == "ellipse":
            if (
                dimensions is None
                or "major_axis" not in dimensions
                or "minor_axis" not in dimensions
            ):
                return {"error": "椭圆计算需要提供major_axis和minor_axis参数"}
            a = dimensions["major_axis"] / 2  # 长半轴
            b = dimensions["minor_axis"] / 2  # 短半轴

            if operation == "area":
                result["area"] = self.pi * a * b
            elif operation == "circumference":
                # 椭圆周长近似公式（拉马努金第二近似）
                h = ((a - b) ** 2) / ((a + b) ** 2)
                result["circumference"] = (
                    self.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
                )
            elif operation == "properties":
                result["area"] = self.pi * a * b
                h = ((a - b) ** 2) / ((a + b) ** 2)
                result["circumference"] = (
                    self.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
                )
                result["major_axis"] = 2 * a
                result["minor_axis"] = 2 * b
                result["eccentricity"] = math.sqrt(1 - (b**2 / a**2)) if a > b else 0

        elif shape_type == "triangle":
            if operation == "area":
                if dimensions and "base" in dimensions and "height" in dimensions:
                    result["area"] = 0.5 * dimensions["base"] * dimensions["height"]
                elif points and len(points) == 3:
                    # 使用坐标计算面积
                    area = abs(
                        0.5
                        * (
                            (points[0][0] * (points[1][1] - points[2][1]))
                            + (points[1][0] * (points[2][1] - points[0][1]))
                            + (points[2][0] * (points[0][1] - points[1][1]))
                        )
                    )
                    result["area"] = area
                else:
                    return {"error": "三角形面积计算需要base+height参数或3个坐标点"}

            elif operation == "perimeter":
                if points and len(points) == 3:
                    p1, p2, p3 = points
                    side_a = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                    side_b = math.sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)
                    side_c = math.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
                    result["perimeter"] = side_a + side_b + side_c
                    result["sides"] = [side_a, side_b, side_c]
                elif dimensions and all(
                    k in dimensions for k in ["side_a", "side_b", "side_c"]
                ):
                    sides = [
                        dimensions["side_a"],
                        dimensions["side_b"],
                        dimensions["side_c"],
                    ]
                    result["perimeter"] = sum(sides)
                    result["sides"] = sides
                else:
                    return {
                        "error": "三角形周长计算需要3个坐标点或side_a, side_b, side_c参数"
                    }

            elif operation == "centroid":
                if points and len(points) == 3:
                    centroid_x = sum(p[0] for p in points) / 3
                    centroid_y = sum(p[1] for p in points) / 3
                    result["centroid"] = [centroid_x, centroid_y]
                else:
                    return {"error": "重心计算需要3个坐标点"}

        elif shape_type == "rectangle":
            if (
                dimensions is None
                or "length" not in dimensions
                or "width" not in dimensions
            ):
                return {"error": "矩形计算需要提供length和width参数"}
            length = dimensions["length"]
            width = dimensions["width"]

            if operation == "area":
                result["area"] = length * width
            elif operation == "perimeter":
                result["perimeter"] = 2 * (length + width)
            elif operation == "diagonal":
                result["diagonal"] = math.sqrt(length**2 + width**2)
            elif operation == "properties":
                result["area"] = length * width
                result["perimeter"] = 2 * (length + width)
                result["diagonal"] = math.sqrt(length**2 + width**2)
                result["aspect_ratio"] = length / width

        elif shape_type == "regular_polygon":
            if (
                dimensions is None
                or "sides" not in dimensions
                or "side_length" not in dimensions
            ):
                return {"error": "正多边形计算需要提供sides和side_length参数"}
            n = int(dimensions["sides"])
            s = dimensions["side_length"]

            if n < 3:
                return {"error": "正多边形边数必须大于等于3"}

            if operation == "area":
                result["area"] = (n * s**2) / (4 * math.tan(self.pi / n))
            elif operation == "perimeter":
                result["perimeter"] = n * s
            elif operation == "properties":
                result["area"] = (n * s**2) / (4 * math.tan(self.pi / n))
                result["perimeter"] = n * s
                result["apothem"] = s / (2 * math.tan(self.pi / n))  # 边心距
                result["circumradius"] = s / (2 * math.sin(self.pi / n))  # 外接圆半径
                result["interior_angle"] = (n - 2) * 180 / n  # 内角度数

        elif shape_type == "polygon":
            if points and operation == "area":
                # 使用鞋带公式计算多边形面积
                n = len(points)
                if n < 3:
                    return {"error": "多边形至少需要3个顶点"}
                area = 0
                for i in range(n):
                    j = (i + 1) % n
                    area += points[i][0] * points[j][1]
                    area -= points[j][0] * points[i][1]
                result["area"] = abs(area) / 2

            elif points and operation == "perimeter":
                n = len(points)
                if n < 3:
                    return {"error": "多边形至少需要3个顶点"}
                perimeter = 0
                for i in range(n):
                    j = (i + 1) % n
                    side = math.sqrt(
                        (points[j][0] - points[i][0]) ** 2
                        + (points[j][1] - points[i][1]) ** 2
                    )
                    perimeter += side
                result["perimeter"] = perimeter

            elif points and operation == "centroid":
                n = len(points)
                if n < 3:
                    return {"error": "多边形至少需要3个顶点"}
                centroid_x = sum(p[0] for p in points) / n
                centroid_y = sum(p[1] for p in points) / n
                result["centroid"] = [centroid_x, centroid_y]
            else:
                return {"error": "多边形计算需要提供坐标点"}

        # 精度处理
        if precision is not None:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    result[key] = round(value, precision)
                elif isinstance(value, list):
                    result[key] = [
                        round(v, precision) if isinstance(v, (int, float)) else v
                        for v in value
                    ]

        return {
            "shape_type": shape_type,
            "operation": operation,
            "result": result,
            "dimensions": dimensions,
            "unit": unit,
        }

    def _solid_geometry(
        self,
        shape_type: str,
        operation: str,
        dimensions: Optional[Dict[str, float]],
        precision: Optional[int] = None,
        unit: str = "default",
    ) -> Dict[str, Any]:
        """立体几何计算"""
        result = {}

        if shape_type == "sphere":
            if dimensions is None or "radius" not in dimensions:
                return {"error": "球体计算需要提供radius参数"}
            r = dimensions["radius"]

            if operation == "volume":
                result["volume"] = (4 / 3) * self.pi * r**3
            elif operation == "surface_area":
                result["surface_area"] = 4 * self.pi * r**2
            elif operation == "properties":
                result["volume"] = (4 / 3) * self.pi * r**3
                result["surface_area"] = 4 * self.pi * r**2
                result["diameter"] = 2 * r
                result["radius"] = r

        elif shape_type == "cube":
            if dimensions is None or "side" not in dimensions:
                return {"error": "立方体计算需要提供side参数"}
            side = dimensions["side"]

            if operation == "volume":
                result["volume"] = side**3
            elif operation == "surface_area":
                result["surface_area"] = 6 * side**2
            elif operation == "diagonal":
                result["space_diagonal"] = side * math.sqrt(3)  # 体对角线
                result["face_diagonal"] = side * math.sqrt(2)  # 面对角线
            elif operation == "properties":
                result["volume"] = side**3
                result["surface_area"] = 6 * side**2
                result["space_diagonal"] = side * math.sqrt(3)
                result["face_diagonal"] = side * math.sqrt(2)
                result["edge_length"] = side

        elif shape_type == "cylinder":
            if (
                dimensions is None
                or "radius" not in dimensions
                or "height" not in dimensions
            ):
                return {"error": "圆柱体计算需要提供radius和height参数"}
            r = dimensions["radius"]
            h = dimensions["height"]

            if operation == "volume":
                result["volume"] = self.pi * r**2 * h
            elif operation == "surface_area":
                result["surface_area"] = 2 * self.pi * r * (r + h)
            elif operation == "properties":
                result["volume"] = self.pi * r**2 * h
                result["surface_area"] = 2 * self.pi * r * (r + h)
                result["lateral_area"] = 2 * self.pi * r * h
                result["base_area"] = self.pi * r**2
                result["radius"] = r
                result["height"] = h

        elif shape_type == "cone":
            if (
                dimensions is None
                or "radius" not in dimensions
                or "height" not in dimensions
            ):
                return {"error": "圆锥体计算需要提供radius和height参数"}
            r = dimensions["radius"]
            h = dimensions["height"]

            if operation == "volume":
                result["volume"] = (1 / 3) * self.pi * r**2 * h
            elif operation == "surface_area":
                slant_height = math.sqrt(r**2 + h**2)
                result["surface_area"] = self.pi * r * (r + slant_height)
            elif operation == "properties":
                slant_height = math.sqrt(r**2 + h**2)
                result["volume"] = (1 / 3) * self.pi * r**2 * h
                result["surface_area"] = self.pi * r * (r + slant_height)
                result["lateral_area"] = self.pi * r * slant_height
                result["base_area"] = self.pi * r**2
                result["slant_height"] = slant_height
                result["radius"] = r
                result["height"] = h

        elif shape_type == "pyramid":
            if (
                dimensions is None
                or "base_area" not in dimensions
                or "height" not in dimensions
            ):
                return {"error": "棱锥计算需要提供base_area和height参数"}
            base_area = dimensions["base_area"]
            h = dimensions["height"]

            if operation == "volume":
                result["volume"] = (1 / 3) * base_area * h
            elif operation == "properties":
                result["volume"] = (1 / 3) * base_area * h
                result["base_area"] = base_area
                result["height"] = h

        elif shape_type == "prism":
            if (
                dimensions is None
                or "base_area" not in dimensions
                or "height" not in dimensions
            ):
                return {"error": "棱柱计算需要提供base_area和height参数"}
            base_area = dimensions["base_area"]
            h = dimensions["height"]
            perimeter = dimensions.get("base_perimeter", 0)

            if operation == "volume":
                result["volume"] = base_area * h
            elif operation == "surface_area":
                if perimeter > 0:
                    result["surface_area"] = 2 * base_area + perimeter * h
                else:
                    return {"error": "表面积计算需要base_perimeter参数"}
            elif operation == "properties":
                result["volume"] = base_area * h
                if perimeter > 0:
                    result["surface_area"] = 2 * base_area + perimeter * h
                    result["lateral_area"] = perimeter * h
                result["base_area"] = base_area
                result["height"] = h

        elif shape_type == "torus":
            if (
                dimensions is None
                or "major_radius" not in dimensions
                or "minor_radius" not in dimensions
            ):
                return {"error": "环面计算需要提供major_radius和minor_radius参数"}
            R = dimensions["major_radius"]  # 大半径
            r = dimensions["minor_radius"]  # 小半径

            if r >= R:
                return {"error": "小半径必须小于大半径"}

            if operation == "volume":
                result["volume"] = 2 * self.pi**2 * R * r**2
            elif operation == "surface_area":
                result["surface_area"] = 4 * self.pi**2 * R * r
            elif operation == "properties":
                result["volume"] = 2 * self.pi**2 * R * r**2
                result["surface_area"] = 4 * self.pi**2 * R * r
                result["major_radius"] = R
                result["minor_radius"] = r

        # 精度处理
        if precision is not None:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    result[key] = round(value, precision)

        return {
            "shape_type": shape_type,
            "operation": operation,
            "result": result,
            "dimensions": dimensions,
            "unit": unit,
        }

    def _analytical_geometry(
        self,
        operation: str,
        points: Optional[List[List[float]]],
        precision: Optional[int] = None,
        unit: str = "default",
    ) -> Dict[str, Any]:
        """解析几何计算"""
        if points is None:
            return {"error": "解析几何计算需要提供坐标点"}

        result = {}

        if operation == "distance":
            if len(points) != 2:
                return {"error": "距离计算需要恰好2个点"}
            p1, p2 = points
            if len(p1) == 2 and len(p2) == 2:  # 2D
                distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            elif len(p1) == 3 and len(p2) == 3:  # 3D
                distance = math.sqrt(
                    (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
                )
            else:
                return {"error": "两点必须具有相同的维度"}
            result["distance"] = distance

        elif operation == "midpoint":
            if len(points) != 2:
                return {"error": "中点计算需要恰好2个点"}
            p1, p2 = points
            if len(p1) != len(p2):
                return {"error": "两点必须具有相同的维度"}
            midpoint = [(p1[i] + p2[i]) / 2 for i in range(len(p1))]
            result["midpoint"] = midpoint

        elif operation == "slope":
            if len(points) != 2:
                return {"error": "斜率计算需要恰好2个点"}
            p1, p2 = points
            if len(p1) != 2 or len(p2) != 2:
                return {"error": "斜率计算仅支持2D点"}
            if p2[0] != p1[0]:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                result["slope"] = slope
                result["angle_degrees"] = math.degrees(math.atan(slope))
            else:
                result["slope"] = "undefined (vertical line)"
                result["angle_degrees"] = 90

        elif operation == "line_equation":
            if len(points) != 2:
                return {"error": "直线方程需要恰好2个点"}
            p1, p2 = points
            if len(p1) != 2 or len(p2) != 2:
                return {"error": "直线方程仅支持2D点"}
            if p2[0] != p1[0]:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                intercept = p1[1] - slope * p1[0]
                result["slope_intercept_form"] = f"y = {slope}x + {intercept}"
                result["slope"] = slope
                result["y_intercept"] = intercept
                # 一般式 Ax + By + C = 0
                A, B, C = -slope, 1, -intercept
                result["general_form"] = f"{A}x + {B}y + {C} = 0"
            else:
                result["equation"] = f"x = {p1[0]}"
                result["general_form"] = f"x - {p1[0]} = 0"

        elif operation == "angle":
            if len(points) != 3:
                return {"error": "角度计算需要恰好3个点"}
            # 计算三点构成的角度
            a, b, c = points
            if not all(len(p) == 2 for p in points):
                return {"error": "角度计算仅支持2D点"}
            ba = [a[0] - b[0], a[1] - b[1]]
            bc = [c[0] - b[0], c[1] - b[1]]

            dot_product = ba[0] * bc[0] + ba[1] * bc[1]
            magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
            magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

            if magnitude_ba > 0 and magnitude_bc > 0:
                cos_angle = dot_product / (magnitude_ba * magnitude_bc)
                cos_angle = max(-1, min(1, cos_angle))  # 确保在[-1, 1]范围内
                angle_rad = math.acos(cos_angle)
                angle_deg = math.degrees(angle_rad)
                result["angle_radians"] = angle_rad
                result["angle_degrees"] = angle_deg
            else:
                return {"error": "无法计算角度，存在重合点"}

        # 精度处理
        if precision is not None:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    result[key] = round(value, precision)
                elif isinstance(value, list):
                    result[key] = [
                        round(v, precision) if isinstance(v, (int, float)) else v
                        for v in value
                    ]

        return {
            "operation": operation,
            "points": points,
            "result": result,
            "unit": unit,
        }
