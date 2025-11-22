#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF论文解析模块
从PDF中提取桥梁VIV实验数据
"""

import re
import logging
from typing import Dict, List, Optional, Any
import numpy as np
from config import (
    CSV_FIELDS, REGEX_PATTERNS, VALIDATION_RULES,
    UNIT_CONVERSIONS, BRIDGE_TYPE_MAPPING,
    STRUCTURE_TYPE_MAPPING, COUNTRY_MAPPING
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFParser:
    """PDF解析器基类"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.text = ""
        self.tables = []

    def extract_text(self) -> str:
        """提取PDF文本"""
        try:
            # 尝试使用PyMuPDF(fitz)
            try:
                import fitz
                doc = fitz.open(self.pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                self.text = text
                logger.info(f"Extracted {len(text)} characters using PyMuPDF")
                return text
            except ImportError:
                logger.warning("PyMuPDF not available, trying pdfplumber")

            # 备选: 使用pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(self.pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                self.text = text
                logger.info(f"Extracted {len(text)} characters using pdfplumber")
                return text
            except ImportError:
                logger.error("Neither PyMuPDF nor pdfplumber available")
                return ""

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""

    def extract_tables(self) -> List[List[List[str]]]:
        """提取PDF表格"""
        try:
            import pdfplumber
            with pdfplumber.open(self.pdf_path) as pdf:
                tables = []
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
                self.tables = tables
                logger.info(f"Extracted {len(tables)} tables")
                return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []


class BridgeDataExtractor:
    """桥梁数据提取器"""

    def __init__(self, text: str, tables: Optional[List] = None):
        self.text = text
        self.tables = tables or []
        self.data = {field: None for field in CSV_FIELDS}

    def extract_value_by_regex(self, pattern: str, text: str, data_type: type = float) -> Optional[Any]:
        """使用正则表达式提取值"""
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                if data_type == float:
                    return float(value)
                elif data_type == int:
                    return int(value)
                else:
                    return value
        except Exception as e:
            logger.debug(f"Regex extraction failed: {str(e)}")
        return None

    def extract_from_text(self) -> Dict:
        """从文本中提取数据"""
        # 提取桥梁名称
        bridge_name_pattern = r'(?:Bridge|桥梁)[:\s]*([A-Za-z\u4e00-\u9fa5\s]+?)(?:\n|,|\.)'
        self.data['BridgeName'] = self.extract_value_by_regex(
            bridge_name_pattern, self.text, str
        )

        # 提取跨度
        span_patterns = [
            r'(?:main span|主跨)[:\s]*([0-9.]+)\s*(?:m|米)',
            r'span[:\s]*([0-9.]+)\s*m'
        ]
        for pattern in span_patterns:
            value = self.extract_value_by_regex(pattern, self.text)
            if value:
                self.data['Span_m'] = value
                break

        # 提取宽度
        width_patterns = [
            r'(?:width|宽度|deck width)[:\s]*([0-9.]+)\s*(?:m|米)',
            r'(?:B|b)\s*=\s*([0-9.]+)\s*m'
        ]
        for pattern in width_patterns:
            value = self.extract_value_by_regex(pattern, self.text)
            if value:
                self.data['Width_m'] = value
                break

        # 提取高度
        height_patterns = [
            r'(?:height|高度|depth|梁高|H)[:\s]*([0-9.]+)\s*(?:m|米)',
            r'(?:H|h)\s*=\s*([0-9.]+)\s*m'
        ]
        for pattern in height_patterns:
            value = self.extract_value_by_regex(pattern, self.text)
            if value:
                self.data['Height_m'] = value
                break

        # 提取频率
        freq_patterns = [
            r'(?:natural frequency|自然频率|基频)[:\s]*([0-9.]+)\s*(?:Hz|赫兹)',
            r'f[:\s]*([0-9.]+)\s*Hz'
        ]
        for pattern in freq_patterns:
            value = self.extract_value_by_regex(pattern, self.text)
            if value:
                self.data['Natural_Freq_Hz'] = value
                break

        # 提取振幅
        amp_patterns = [
            r'(?:maximum amplitude|最大振幅|max amplitude)[:\s]*([0-9.]+)\s*(?:mm|毫米)',
            r'amplitude[:\s]*([0-9.]+)\s*mm'
        ]
        for pattern in amp_patterns:
            value = self.extract_value_by_regex(pattern, self.text)
            if value:
                self.data['Max_Amplitude_mm'] = value
                break

        # 提取阻尼比
        damping_patterns = [
            r'(?:damping ratio|阻尼比)[:\s]*([0-9.]+)\s*%',
            r'(?:ζ|zeta)[:\s]*([0-9.]+)\s*%'
        ]
        for pattern in damping_patterns:
            value = self.extract_value_by_regex(pattern, self.text)
            if value:
                self.data['Damping_Ratio'] = value / 100  # 转换为小数
                break

        # 提取风速
        wind_patterns = [
            r'(?:critical wind speed|临界风速)[:\s]*([0-9.]+)\s*(?:m/s|米/秒)',
            r'(?:U|u)[:\s]*([0-9.]+)\s*m/s'
        ]
        for pattern in wind_patterns:
            value = self.extract_value_by_regex(pattern, self.text)
            if value:
                self.data['Critical_Wind_Speed_ms'] = value
                break

        return self.data

    def extract_from_tables(self) -> Dict:
        """从表格中提取数据"""
        if not self.tables:
            return self.data

        for table in self.tables:
            if not table or len(table) < 2:
                continue

            # 尝试识别表头
            header = [str(cell).lower() for cell in table[0]]

            # 查找关键列
            span_col = self._find_column(header, ['span', '跨度', 'main span'])
            width_col = self._find_column(header, ['width', '宽度', 'b'])
            height_col = self._find_column(header, ['height', '高度', 'h', 'depth'])
            freq_col = self._find_column(header, ['frequency', '频率', 'f'])
            amp_col = self._find_column(header, ['amplitude', '振幅'])
            damping_col = self._find_column(header, ['damping', '阻尼'])

            # 提取数据行
            for row in table[1:]:
                if len(row) < 2:
                    continue

                try:
                    if span_col is not None and not self.data['Span_m']:
                        self.data['Span_m'] = self._parse_number(row[span_col])

                    if width_col is not None and not self.data['Width_m']:
                        self.data['Width_m'] = self._parse_number(row[width_col])

                    if height_col is not None and not self.data['Height_m']:
                        self.data['Height_m'] = self._parse_number(row[height_col])

                    if freq_col is not None and not self.data['Natural_Freq_Hz']:
                        self.data['Natural_Freq_Hz'] = self._parse_number(row[freq_col])

                    if amp_col is not None and not self.data['Max_Amplitude_mm']:
                        self.data['Max_Amplitude_mm'] = self._parse_number(row[amp_col])

                    if damping_col is not None and not self.data['Damping_Ratio']:
                        value = self._parse_number(row[damping_col])
                        if value and value > 0.1:  # 如果是百分比
                            value = value / 100
                        self.data['Damping_Ratio'] = value

                except Exception as e:
                    logger.debug(f"Error parsing row: {str(e)}")
                    continue

        return self.data

    def _find_column(self, header: List[str], keywords: List[str]) -> Optional[int]:
        """在表头中查找包含关键词的列"""
        for i, col in enumerate(header):
            for keyword in keywords:
                if keyword in col.lower():
                    return i
        return None

    def _parse_number(self, text: Any) -> Optional[float]:
        """从文本中解析数字"""
        if text is None:
            return None
        try:
            # 移除非数字字符(保留小数点)
            text = str(text).strip()
            number_match = re.search(r'([0-9.]+)', text)
            if number_match:
                return float(number_match.group(1))
        except Exception:
            pass
        return None

    def calculate_derived_fields(self):
        """计算衍生字段"""
        # 计算宽高比
        if self.data['Width_m'] and self.data['Height_m'] and self.data['Height_m'] > 0:
            self.data['Width_Height_Ratio'] = self.data['Width_m'] / self.data['Height_m']

    def validate_data(self) -> bool:
        """验证数据合理性"""
        valid = True

        for field, value in self.data.items():
            if value is None:
                continue

            if field in VALIDATION_RULES:
                min_val, max_val = VALIDATION_RULES[field]
                if not (min_val <= value <= max_val):
                    logger.warning(
                        f"Field {field} value {value} out of range [{min_val}, {max_val}]"
                    )
                    valid = False

        # 检查必填字段
        if not self.data['BridgeName'] or not self.data['Max_Amplitude_mm']:
            logger.warning("Missing required fields: BridgeName or Max_Amplitude_mm")
            valid = False

        return valid


def parse_pdf(pdf_path: str, paper_info: Optional[Dict] = None) -> Dict:
    """
    解析PDF文件,提取桥梁数据

    Args:
        pdf_path: PDF文件路径
        paper_info: 论文信息(来自搜索模块)

    Returns:
        提取的桥梁数据字典
    """
    logger.info(f"Parsing PDF: {pdf_path}")

    # 提取文本和表格
    parser = PDFParser(pdf_path)
    text = parser.extract_text()
    tables = parser.extract_tables()

    # 提取数据
    extractor = BridgeDataExtractor(text, tables)
    data = extractor.extract_from_text()
    data = extractor.extract_from_tables()
    extractor.calculate_derived_fields()

    # 添加论文元数据
    if paper_info:
        data['PaperSource'] = paper_info.get('title', 'Unknown')
        data['Year'] = paper_info.get('year', None)

    # 验证数据
    is_valid = extractor.validate_data()
    logger.info(f"Data validation: {'PASS' if is_valid else 'FAIL'}")

    return data


def parse_manual_data(bridge_info: Dict) -> Dict:
    """
    手动输入的桥梁数据解析
    用于从已知数据源(如表格、数据库)导入数据

    Args:
        bridge_info: 包含桥梁信息的字典

    Returns:
        标准化后的桥梁数据
    """
    data = {field: None for field in CSV_FIELDS}

    # 复制已有字段
    for key, value in bridge_info.items():
        if key in data:
            data[key] = value

    # 创建提取器进行验证和衍生字段计算
    extractor = BridgeDataExtractor("", [])
    extractor.data = data
    extractor.calculate_derived_fields()
    extractor.validate_data()

    return extractor.data


if __name__ == "__main__":
    # 测试手动数据解析
    print("Testing Manual Data Parsing")
    print("=" * 50)

    # 示例数据
    sample_bridge = {
        'BridgeName': 'Test Bridge',
        'Country': 'China',
        'BridgeType': 'Suspension',
        'Span_m': 1000,
        'Width_m': 35.0,
        'Height_m': 3.0,
        'Natural_Freq_Hz': 0.25,
        'Damping_Ratio': 0.015,
        'Max_Amplitude_mm': 45.5,
        'PaperSource': 'Test Data'
    }

    parsed_data = parse_manual_data(sample_bridge)

    print("\nParsed Data:")
    for key, value in parsed_data.items():
        if value is not None:
            print(f"  {key}: {value}")
