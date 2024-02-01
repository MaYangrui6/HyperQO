import re
from string import digits

import re
from string import digits

class BagOfPredicates:
    def __init__(self):
        # 定义字符替换规则
        self.replaces = [(" ", ""), ("(", ""), (")", ""), ("[", ""), ("]", ""), ("::text", ""),
                         ("::bpchar", ""), ("::numeric", ""), ("::date", ""), ("::integer", ""), ("::timestampwithouttimezone", "")]
        self.remove_digits = str.maketrans("", "", digits)

        # 定义感兴趣的操作符
        self.INTERESTING_OPERATORS = [
            "Aggregate",
            'CTE Scan',
            'Hash Join',
            'Index Only Scan',
            'Index Scan',
            'Merge Join',
            'Nested Loop',
            'Seq Scan',
            'Sort',
        ]
        self.relevant_operators = None

    def extract_predicates_from_plan(self, plan):
        """从执行计划中提取感兴趣的谓词信息"""
        self.relevant_operators = []
        self._parse_plan(plan)
        return self.relevant_operators

    def _parse_plan(self, plan):
        """递归解析执行计划"""
        node_type = plan["Node Type"]
        if node_type in self.INTERESTING_OPERATORS:
            contains_necessity = any(element in plan.keys() for element in ["Filter", "Cond"])
            if contains_necessity:
                node_representation = self._parse_node(plan)
                self.relevant_operators.append(node_representation)
        if "Plans" not in plan:
            return
        for sub_plan in plan["Plans"]:
            self._parse_plan(sub_plan)

    def _parse_node(self, node):
        """解析单个节点"""
        node_type_repr = f"{node['Node Type'].replace(' ', '_')}_"
        parse_method = getattr(self, f"_parse_{node['Node Type'].replace(' ', '_').lower()}", self._parse_unknown)
        return node_type_repr + parse_method(node)

    def _parse_unknown(self, node):
        """处理未知节点类型"""
        return "Unknown"

    def _parse_attributes(self, node, attributes):
        """解析节点的属性信息"""
        representation = [self._stringify_attribute_columns(node, attribute) for attribute in attributes]
        return ''.join(representation)

    def _stringify_attribute_columns(self, node, attribute):
        """将节点的属性转换为字符串表示"""
        attribute_repr = f"{attribute.replace(' ', '_')}_"
        if attribute in node:
            value = node[attribute]
            value = self._apply_replacements(value)
            value = re.sub(r'".*?"', "", value)
            value = re.sub(r"'.*?'", "", value)
            value = value.translate(self.remove_digits)
            attribute_repr += value
        return attribute_repr

    def _apply_replacements(self, value):
        """应用字符替换规则"""
        if isinstance(value, list):
            return ''.join([self._apply_replacements_single(elem) for elem in value])
        else:
            return self._apply_replacements_single(value)

    def _apply_replacements_single(self, value):
        """对单个元素应用字符替换规则"""
        for replace, replacement in self.replaces:
            value = value.replace(replace, replacement)
        return value

    def _parse_seq_scan(self, node):
        """解析 Seq Scan 节点"""
        assert "Relation Name" in node
        node_repr = f"{node['Relation Name']}_"
        node_repr += self._parse_attributes(node, ["Filter"])
        return node_repr

    def _parse_index_scan(self, node):
        """解析 Index Scan 节点"""
        assert "Relation Name" in node
        node_repr = f"{node['Relation Name']}_"
        node_repr += self._parse_attributes(node, ["Filter", "Index Cond"])
        return node_repr

    def _parse_index_only_scan(self, node):
        """解析 Index Only Scan 节点"""
        assert "Relation Name" in node
        node_repr = f"{node['Relation Name']}_"
        node_repr += self._parse_attributes(node, ["Index Cond"])
        return node_repr

    def _parse_cte_scan(self, node):
        """解析 CTE Scan 节点"""
        assert "CTE Name" in node
        node_repr = f"{node['CTE Name']}_"
        node_repr += self._parse_attributes(node, ["Filter"])
        return node_repr

    def _parse_nested_loop(self, node):
        """解析 Nested Loop 节点"""
        return self._parse_attributes(node, ["Join Filter"])

    def _parse_hash_join(self, node):
        """解析 Hash Join 节点"""
        return self._parse_attributes(node, ["Join Filter", "Hash Cond"])

    def _parse_merge_join(self, node):
        """解析 Merge Join 节点"""
        return self._parse_attributes(node, ["Merge Cond"])

    def _parse_aggregate(self, node):
        """解析 Aggregate 节点"""
        return self._parse_attributes(node, ["Filter", "Group Key"])

    def _parse_sort(self, node):
        """解析 Sort 节点"""
        return self._parse_attributes(node, ["Sort Key"])


