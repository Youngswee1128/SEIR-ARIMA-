from __future__ import annotations

import argparse
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def wtag(name: str) -> str:
    return f"{{{NS['w']}}}{name}"


def paragraph_text(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        if node.tag == wtag("t") and node.text:
            parts.append(node.text)
        elif node.tag == wtag("tab"):
            parts.append("\t")
        elif node.tag == wtag("br"):
            parts.append("\n")
    return "".join(parts)


def clear_paragraph_keep_ppr(p: ET.Element) -> ET.Element | None:
    ppr = p.find("w:pPr", NS)
    preserved = deepcopy(ppr) if ppr is not None else None
    for child in list(p):
        p.remove(child)
    if preserved is not None:
        p.append(preserved)
    return preserved


def ensure_ppr(p: ET.Element) -> ET.Element:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        ppr = ET.Element(wtag("pPr"))
        p.insert(0, ppr)
    return ppr


def set_alignment(p: ET.Element, val: str) -> None:
    ppr = ensure_ppr(p)
    old = ppr.find("w:jc", NS)
    if old is not None:
        ppr.remove(old)
    jc = ET.SubElement(ppr, wtag("jc"))
    jc.set(wtag("val"), val)


def set_spacing_after(p: ET.Element, after: str = "120") -> None:
    ppr = ensure_ppr(p)
    spacing = ppr.find("w:spacing", NS)
    if spacing is None:
        spacing = ET.SubElement(ppr, wtag("spacing"))
    spacing.set(wtag("after"), after)


def append_text_run(
    p: ET.Element,
    text: str,
    *,
    bold: bool = False,
    size: str | None = None,
) -> None:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if i:
            r_br = ET.SubElement(p, wtag("r"))
            ET.SubElement(r_br, wtag("br"))
        r = ET.SubElement(p, wtag("r"))
        if bold or size:
            rpr = ET.SubElement(r, wtag("rPr"))
            if bold:
                ET.SubElement(rpr, wtag("b"))
            if size:
                sz = ET.SubElement(rpr, wtag("sz"))
                sz.set(wtag("val"), size)
                sz_cs = ET.SubElement(rpr, wtag("szCs"))
                sz_cs.set(wtag("val"), size)
        t = ET.SubElement(r, wtag("t"))
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = line


def replace_text(
    p: ET.Element,
    text: str,
    *,
    bold: bool = False,
    size: str | None = None,
    align: str | None = None,
    after: str | None = None,
) -> None:
    clear_paragraph_keep_ppr(p)
    if align:
        set_alignment(p, align)
    if after:
        set_spacing_after(p, after)
    append_text_run(p, text, bold=bold, size=size)


def make_paragraph(
    text: str,
    *,
    bold: bool = False,
    size: str | None = None,
    align: str | None = None,
    after: str | None = "120",
) -> ET.Element:
    p = ET.Element(wtag("p"))
    if align:
        set_alignment(p, align)
    if after:
        set_spacing_after(p, after)
    append_text_run(p, text, bold=bold, size=size)
    return p


def make_intro_paragraphs() -> list[ET.Element]:
    intro = [
        make_paragraph("第1节 引言（Introduction）", bold=True, size="32", after="180"),
        make_paragraph(
            "呼吸道传染病由于传播速度快、传播范围广，对全球公共卫生和社会经济稳定构成持续挑战。"
            "历史上由流感病毒、SARS冠状病毒以及新型冠状病毒引发的大规模疫情表明，此类疾病不仅会造成显著的健康负担，"
            "还会通过医疗资源挤兑、人员流动受限和社会行为改变影响公共治理体系。"
            "因此，对呼吸道传染病传播过程进行动态建模与短期预测，是流行病学研究和公共卫生决策中的重要问题"
            "（Asplin et al., 2024; Chen et al., 2021; Vos et al., 2020）。",
        ),
        make_paragraph(
            "在众多呼吸道疾病中，甲型流感和COVID-19具有较强代表性。甲型流感具有遗传变异和跨宿主传播特征，"
            "容易形成季节性流行甚至全球性大流行（Taubenberger & Kash, 2010）；COVID-19的传播过程则受到人群流动、"
            "非药物干预措施和行为变化等因素共同影响，呈现明显的非平稳性和阶段性（World Health Organization, 2020; "
            "Karia et al., 2020; Chinazzi et al., 2020; Flaxman et al., 2020）。"
            "这两类疾病的时间演化特征为机制建模和时间序列预测提供了典型应用场景。",
        ),
        make_paragraph(
            "分区流行病动力学模型因具有明确的流行病学解释，被广泛用于刻画疾病传播机制。"
            "经典SIR模型由Kermack和McKendrick（1927）提出，为现代传染病建模奠定了基础；"
            "在此基础上引入潜伏期阶段形成的SEIR模型，能够更合理地描述具有潜伏期特征的呼吸道传染病传播过程。"
            "针对死亡风险较高的疫情，还可进一步扩展为SEIRD模型以纳入死亡仓室。已有研究表明，SEIR及其扩展模型仍是当前传染病建模的重要工具"
            "（Tang et al., 2020; Eshtewy et al., 2026）。",
        ),
        make_paragraph(
            "与此同时，统计时间序列方法也被广泛用于传染病短期预测。自回归积分滑动平均模型（ARIMA）能够捕捉历史病例序列中的线性趋势和时间依赖结构，"
            "在中短期预测中具有较强稳定性（Box & Jenkins, 1970）。例如，Benvenuto等（2020）使用ARIMA模型预测COVID-19流行趋势；"
            "He和Tao（2018）基于监测数据构建ARIMA/SARIMA模型预测流感病毒阳性率；Tsan等（2022）比较ARIMA与LSTM在呼吸道疾病预测中的表现，"
            "发现ARIMA在中短期场景下具有稳定的拟合能力。然而，纯统计模型主要依赖历史数据模式，缺乏对传播机制的解释能力，在长期预测和情景分析中存在局限"
            "（Hyndman & Athanasopoulos, 2018）。",
        ),
        make_paragraph(
            "鉴于机制模型与统计模型各有优势，近年来研究者开始探索混合建模框架。该类方法将流行病动力学模型对传播过程的解释能力，"
            "与数据驱动方法对复杂时间序列波动的拟合能力相结合，从而在保持机制合理性的同时提升预测精度。"
            "Zhang（2003）提出将ARIMA与神经网络结合以同时刻画线性和非线性特征；Shaman和Karspeck（2012）、Pei和Shaman（2020）"
            "进一步展示了传播动力学模型与数据同化或统计校正方法结合在流感和COVID-19预测中的有效性。",
        ),
        make_paragraph(
            "基于上述研究脉络，本文构建SEIR/SEIRD机制模型与ARIMA残差修正相结合的混合预测框架，并引入MCMC方法进行参数估计，"
            "以增强参数估计的稳健性并量化不确定性。与仅基于单一疫情数据或整体时间序列的研究不同，本文同时选取2009年甲型H1N1流感和COVID-19初期数据进行验证，"
            "并通过分段建模刻画不同传播阶段的参数变化。本文的目标是在统一框架下兼顾模型解释性、短期预测精度和跨病种适用性，为呼吸道传染病传播预测提供方法支持。",
        ),
    ]
    return intro


REPLACEMENTS = {
    3: ("第2节 数据与方法（Materials and Methods）", True, "30"),
    9: (
        "由于疫情传播过程通常具有明显的阶段性特征，防控政策变化、人群行为调整以及病毒变异等因素均可能导致传播参数随时间变化。"
        "单一固定参数模型难以对完整时间序列进行精确拟合。为提高模型灵活性并增强其对复杂动态过程的刻画能力，本文引入分段建模策略。",
        False,
        None,
    ),
    11: (
        "对于COVID-19数据，考虑到样本期较短且疫情处于早期快速变化阶段，本文在整体SEIRD框架下拟合传播过程，并将最后7天作为预测集用于模型验证。",
        False,
        None,
    ),
    16: (
        "整体方法框架如图1所示，主要包括四个阶段：\n"
        "（1）基于SEIR/SEIRD模型对疫情传播趋势建模；\n"
        "（2）利用MCMC方法进行参数估计；\n"
        "（3）计算机制模型残差，并使用ARIMA模型对残差序列进行建模与预测；\n"
        "（4）叠加机制模型预测值与ARIMA残差预测值，评估SEIR/SEIRD+ARIMA混合模型的预测能力。",
        False,
        None,
    ),
    39: (
        "本文采用基于贝叶斯推断的马尔可夫链蒙特卡罗（Markov Chain Monte Carlo，MCMC）方法估计模型参数和初始条件。"
        "MCMC是贝叶斯统计计算中常用的随机采样方法，能够通过构造以目标后验分布为平稳分布的马尔可夫链，获得参数后验分布的近似样本（Gilks et al., 1995）。"
        "在传染病动力学模型中，MCMC适用于处理非线性参数关系和高维参数空间，也便于量化参数估计不确定性。",
        False,
        None,
    ),
    40: (
        "具体而言，本文在给定观测病例数据和模型结构的条件下，对传播率、初始状态及相关待估参数进行后验采样，并根据采样结果得到参数点估计和不确定性范围。"
        "这一处理方式有助于避免单次确定性优化对初始值和局部最优的敏感性。",
        False,
        None,
    ),
    41: (
        "相比传统最小二乘等点估计方法，MCMC不仅能够获得较优的参数估计值，还能够进一步反映参数之间的相关性及其不确定性。"
        "因此，在本文所涉及的分段SEIR/SEIRD模型中，MCMC为后续残差建模和预测误差评估提供了更稳健的参数基础。",
        False,
        None,
    ),
    53: ("其中，B为滞后算子，ε_t为白噪声序列，d表示差分阶数。", False, None),
    59: (
        "该混合模型的优势在于：SEIR/SEIRD模型提供长期传播趋势，ARIMA模型修正短期残差波动，两者结合能够提高短期预测精度并降低单一机制模型的系统性偏差。",
        False,
        None,
    ),
    66: (
        "模型预测性能采用平均绝对误差（MAE）、预测终点绝对误差（endpoint_abs_error）、均方根误差（RMSE）和加权绝对百分比误差（WAPE）进行评价。"
        "MAE和RMSE衡量整体误差水平，endpoint_abs_error衡量预测末端偏差，WAPE反映预测误差占实际病例规模的比例。",
        False,
        None,
    ),
    75: ("这些指标从整体误差、终点偏差和相对误差三个角度评估模型预测的准确性与稳定性。", False, None),
    77: ("第3节 结果（Results）", True, "30"),
    87: (
        "基于上述参数建立的分段SEIR模型对H1N1数据进行了拟合与预测。以第一阶段14天预测结果为例，其拟合效果如图2所示。",
        False,
        None,
    ),
    90: ("图2 H1N1第一阶段SEIR模型拟合与14天预测结果", False, None),
    92: (
        "从图2可以看出，SEIR模型能够较好刻画第一阶段累计病例的增长趋势，模型预测曲线与真实病例变化趋势整体保持一致，说明该模型能够有效反映H1N1疫情的长期传播规律。",
        False,
        None,
    ),
    100: (
        "在完成SEIR模型拟合与ARIMA残差预测后，本文进一步构建SEIR+ARIMA混合模型。该模型将SEIR模型预测值与ARIMA预测残差进行叠加，从而同时兼顾疫情传播长期趋势与短期随机波动。"
        "SEIR+ARIMA混合模型在两个阶段的预测结果分别如图3和图4所示。",
        False,
        None,
    ),
    102: ("图3 H1N1第一阶段SEIR+ARIMA混合模型预测结果", False, None),
    104: ("图4 H1N1第二阶段SEIR+ARIMA混合模型预测结果", False, None),
    109: ("3.1.3 SEIR+ARIMA混合模型与单一SEIR模型误差对比", False, None),
    110: ("通过计算预测的日新增病例数和累计病例数与实际观测值之间的误差，得到模型评价指标，结果如表3和表4所示。", False, None),
    112: ("表3、表4 H1N1不同模型预测误差对比", False, None),
    113: (
        "从表3和表4可以看出，SEIR+ARIMA混合模型在各指标上均明显优于单一SEIR模型，说明引入ARIMA残差修正能够有效提升短期预测能力，提高模型预测精度与稳定性，但不同阶段的改进幅度存在一定差异。",
        False,
        None,
    ),
    114: (
        "对于日新增病例预测，第一阶段中SEIR+ARIMA模型的MAE、WAPE和RMSE均较单一SEIR模型有所下降，其中MAE降低了33.8%。"
        "第二阶段改进更加显著，MAE由477.80降低至88.48，下降幅度达到81.5%，说明ARIMA能够有效修正疫情复杂波动阶段的预测误差。",
        False,
        None,
    ),
    122: (
        "针对COVID-19疫情数据，本文采用SEIRD模型对疫情整体传播过程进行建模，并利用MCMC方法完成模型参数估计。"
        "随后按照相同步骤构建SEIRD+ARIMA组合模型，其预测结果如图5所示。",
        False,
        None,
    ),
    125: ("图5 COVID-19 SEIRD+ARIMA混合模型预测结果（左图为日新增病例）", False, None),
    126: (
        "SEIRD模型能够较好反映COVID-19疫情初期的整体传播趋势，并刻画其长期动力学特征，但在预测阶段对日新增病例波动变化的刻画能力有限。"
        "引入ARIMA残差修正后，SEIRD+ARIMA混合模型能够进一步捕捉时间序列中的短期相关性与随机扰动特征，使预测结果更加接近实际观测数据。"
        "从累计病例预测结果看，混合模型还能够减弱机理模型在长期预测中的误差累积，从而提高累计病例终点预测精度。",
        False,
        None,
    ),
    129: ("通过计算预测的累计病例数与实际观测值之间的误差，得到模型评价指标，结果如表5和表6所示。", False, None),
    131: (
        "由表5和表6可以看出，引入ARIMA残差修正后，SEIRD模型的预测性能整体得到提升。"
        "对于日新增病例预测，SEIRD+ARIMA模型的MAE、WAPE和RMSE均明显低于单一SEIRD模型，其中MAE下降了40.4%，表明混合模型能够更有效地刻画新增病例序列中的短期波动特征。"
        "对于累计预测结果，SEIRD+ARIMA模型同样表现出更高精度。病例累计预测中MAE降低了45.4%，终点绝对误差明显减小；死亡累计预测中MAE降低了62.2%，说明残差修正能够有效减弱长期预测中的误差累积，提高模型终点预测能力。",
        False,
        None,
    ),
    133: ("第4节 讨论（Discussion）", True, "30"),
    142: (
        "尽管本文方法取得了较好的预测效果，但仍存在一定局限性。首先，SEIR/SEIRD模型中的参数在不同阶段可能随防控措施、人口行为以及病毒变异而动态变化，"
        "而本文主要采用阶段性固定参数进行建模，仍难以完全反映真实传播过程中的连续变化。其次，ARIMA本质上属于线性时间序列模型，对于复杂非线性波动和突发性结构变化的刻画能力有限。"
        "此外，本文主要基于病例与死亡数据进行分析，尚未进一步引入人口流动、气象因素以及社会干预措施等多源数据。",
        False,
        None,
    ),
}


DELETE_PARAGRAPHS = {
    1,
    2,
    60,
    61,
    62,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    88,
    123,
}


def revise_docx(base_docx: Path, out_docx: Path) -> None:
    with zipfile.ZipFile(base_docx) as zin:
        xml = zin.read("word/document.xml")
        root = ET.fromstring(xml)
        body = root.find("w:body", NS)
        if body is None:
            raise RuntimeError("word/document.xml has no body")

        paragraphs = [el for el in body if el.tag == wtag("p")]
        p_by_idx = {idx: p for idx, p in enumerate(paragraphs, start=1)}

        for idx, (text, bold, size) in REPLACEMENTS.items():
            if idx not in p_by_idx:
                raise RuntimeError(f"paragraph {idx} not found")
            align = "center" if idx in {90, 102, 104, 112, 125} else None
            replace_text(p_by_idx[idx], text, bold=bold, size=size, align=align, after="120")

        # Insert a caption immediately after the framework diagram.
        framework_p = p_by_idx[17]
        framework_caption = make_paragraph(
            "图1 SEIR/SEIRD+ARIMA混合模型方法框架",
            align="center",
            after="120",
        )
        body.insert(list(body).index(framework_p) + 1, framework_caption)

        for idx in sorted(DELETE_PARAGRAPHS, reverse=True):
            p = p_by_idx.get(idx)
            if p is not None and p in list(body):
                body.remove(p)

        first_child = next(iter(body))
        intro_paragraphs = make_intro_paragraphs()
        for offset, p in enumerate(intro_paragraphs):
            body.insert(offset, p)
        body.insert(len(intro_paragraphs), make_paragraph("", after="120"))

        document_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        tmp = out_docx.with_suffix(".tmp.docx")
        with zipfile.ZipFile(base_docx) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr(item, document_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))
        if out_docx.exists():
            out_docx.unlink()
        shutil.move(str(tmp), str(out_docx))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    revise_docx(args.base, args.out)


if __name__ == "__main__":
    main()
