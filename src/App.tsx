import { useState, useMemo, useEffect } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Scatter,
  Line,
} from "recharts";
import {
  TrendingUp,
  Sliders,
  Database,
  Download,
  Copy,
  RotateCcw,
  Sparkles,
  Check,
  FileText,
  AlertCircle,
  HelpCircle,
  Info,
} from "lucide-react";
import { motion } from "motion/react";

// Define TypeScript Interfaces
interface DataPoint {
  index: number;
  x: number;
  y: number;
  yHat: number;
  residual: number;
}

interface DatasetResult {
  points: DataPoint[];
  m: number;
  c: number;
  actualR2: number;
  description: string;
  xMean: number;
  yMean: number;
  xMinActual: number;
  xMaxActual: number;
  yMinActual: number;
  yMaxActual: number;
}

export default function App() {
  // 1. App State for Configuration Parameters
  const [xMin, setXMin] = useState<number>(10);
  const [xMax, setXMax] = useState<number>(100);
  const [yMin, setYMin] = useState<number>(100);
  const [yMax, setYMax] = useState<number>(1000);
  const [targetR2, setTargetR2] = useState<number>(0.85);
  const [sampleSize, setSampleSize] = useState<number>(50);
  const [seed, setSeed] = useState<number>(1); // To trigger random permutations

  // Notification and transient states
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // 2. Validate Inputs dynamically
  useEffect(() => {
    if (xMin >= xMax) {
      setValidationError("自变量区间错误：最小值 (Min) 必须严格小于最大值 (Max)。已自动采用默认间距进行拟合。");
    } else if (yMin >= yMax) {
      setValidationError("因变量规模错误：数值规模最小值 (Min) 必须严格小于最大值 (Max)。");
    } else {
      setValidationError(null);
    }
  }, [xMin, xMax, yMin, yMax]);

  // Show floating toast message helper
  const triggerToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(null);
    }, 2500);
  };

  // Preset Handlers
  const applyPreset = (presetType: "perfect" | "strong" | "moderate" | "weak" | "zero") => {
    setSeed((prev) => prev + 1); // Trigger new randomized distribution
    switch (presetType) {
      case "perfect":
        setTargetR2(1.0);
        setXMin(10);
        setXMax(100);
        setYMin(100);
        setYMax(1000);
        setSampleSize(40);
        triggerToast("已加载：💎 极优完美拟合模型 (R² = 1.00)");
        break;
      case "strong":
        setTargetR2(0.92);
        setXMin(20);
        setXMax(120);
        setYMin(500);
        setYMax(2500);
        setSampleSize(60);
        triggerToast("已加载：📈 金融/产业强正相关模型 (R² = 0.92)");
        break;
      case "moderate":
        setTargetR2(0.65);
        setXMin(0);
        setXMax(50);
        setYMin(10);
        setYMax(100);
        setSampleSize(75);
        triggerToast("已加载：📊 自然科学中等干扰模型 (R² = 0.65)");
        break;
      case "weak":
        setTargetR2(0.25);
        setXMin(100);
        setXMax(500);
        setYMin(1000);
        setYMax(5000);
        setSampleSize(100);
        triggerToast("已加载：🌪️ 高干扰/弱正相关模型 (R² = 0.25)");
        break;
      case "zero":
        setTargetR2(0.0);
        setXMin(5);
        setXMax(95);
        setYMin(50);
        setYMax(500);
        setSampleSize(50);
        triggerToast("已加载：🎲 纯白噪声/零相关模型 (R² = 0.00)");
        break;
    }
  };

  // 3. Dataset Generation Engine (strictly fits constraints via Gram-Schmidt orthogonalization)
  const datasetResult = useMemo<DatasetResult>(() => {
    // Standard fallbacks if bounds are corrupted
    const activeXMin = xMin;
    const activeXMax = xMin >= xMax ? xMin + 10 : xMax;
    const activeYMin = yMin;
    const activeYMax = yMin >= yMax ? yMin + 100 : yMax;
    const n = Math.max(5, Math.min(300, sampleSize));
    const target = Math.max(0, Math.min(1.0, targetR2));

    // LCG pseudo-random generator with seed to ensure reproducible/controlled variance
    let seedValue = seed;
    const rng = () => {
      seedValue = (seedValue * 1664525 + 1013904223) % 4294967296;
      return seedValue / 4294967296;
    };

    // A. Generate X values strictly bounded in [activeXMin, activeXMax]
    // To ensure full coverage of the range, we pin the first and last element, and sort them.
    const x = new Array(n);
    x[0] = activeXMin;
    x[n - 1] = activeXMax;
    for (let i = 1; i < n - 1; i++) {
      x[i] = activeXMin + rng() * (activeXMax - activeXMin);
    }
    x.sort((a, b) => a - b);

    // B. Calculate Mean & Standard Deviation of X
    const xSum = x.reduce((s, val) => s + val, 0);
    const meanX = xSum / n;
    const varX = x.reduce((s, val) => s + Math.pow(val - meanX, 2), 0) / n;
    const stdX = Math.sqrt(varX);

    // If stdX is 0, return fallback
    if (stdX === 0) {
      return {
        points: [],
        m: 0,
        c: 0,
        actualR2: 0,
        description: "自变量区间无效，无法生成拟合。",
        xMean: 0,
        yMean: 0,
        xMinActual: activeXMin,
        xMaxActual: activeXMax,
        yMinActual: activeYMin,
        yMaxActual: activeYMax,
      };
    }

    // Standardize X -> X' (mean 0, variance 1)
    const xPrime = x.map((val) => (val - meanX) / stdX);

    // C. Generate raw random noise vector Z using Box-Muller transform for normal distribution look
    const z = new Array(n);
    for (let i = 0; i < n; i++) {
      const u1 = rng() || 0.0001;
      const u2 = rng();
      const normalVal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      z[i] = normalVal;
    }

    // Standardize Z -> Z' (mean 0, variance 1)
    const zMean = z.reduce((s, val) => s + val, 0) / n;
    const zVar = z.reduce((s, val) => s + Math.pow(val - zMean, 2), 0) / n;
    const zStd = Math.sqrt(zVar) || 1;
    const zPrime = z.map((val) => (val - zMean) / zStd);

    // D. Gram-Schmidt Orthogonalization: make Z' orthogonal to X'
    // e_i = Z'_i - Cov(X', Z') * X'_i
    let covXZ = 0;
    for (let i = 0; i < n; i++) {
      covXZ += xPrime[i] * zPrime[i];
    }
    covXZ /= n;

    const e = new Array(n);
    for (let i = 0; i < n; i++) {
      e[i] = zPrime[i] - covXZ * xPrime[i];
    }

    // Standardize Residual e -> e' (mean 0, variance 1, orthogonal to xPrime)
    const eMean = e.reduce((s, val) => s + val, 0) / n;
    const eVar = e.reduce((s, val) => s + Math.pow(val - eMean, 2), 0) / n;
    const eStd = Math.sqrt(eVar) || 1;
    const ePrime = e.map((val) => (val - eMean) / eStd);

    // E. Synthesize standardized target variable: Y' = r * X' + sqrt(1 - r^2) * e'
    // where r = sqrt(targetR2)
    const r = Math.sqrt(target);
    const coefE = Math.sqrt(1 - target);

    const yPrime = new Array(n);
    for (let i = 0; i < n; i++) {
      yPrime[i] = r * xPrime[i] + coefE * ePrime[i];
    }

    // F. Linearly scale standardized Y' to match user's custom scale [activeYMin, activeYMax]
    // This is a positive linear scaling: y_i = A * yPrime_i + B, where A > 0.
    // This maintains the exact correlation with X' (and thus X) perfectly!
    const minYPrime = Math.min(...yPrime);
    const maxYPrime = Math.max(...yPrime);
    const rangeYPrime = maxYPrime - minYPrime || 1;

    const y = new Array(n);
    for (let i = 0; i < n; i++) {
      y[i] = activeYMin + ((yPrime[i] - minYPrime) / rangeYPrime) * (activeYMax - activeYMin);
    }

    // G. Recalculate optimal linear regression parameters (m, c) on final x, y to verify
    const ySum = y.reduce((s, val) => s + val, 0);
    const meanY = ySum / n;

    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      num += (x[i] - meanX) * (y[i] - meanY);
      den += Math.pow(x[i] - meanX, 2);
    }
    const m = den === 0 ? 0 : num / den;
    const c = meanY - m * meanX;

    // H. Calculate predicted values (yHat), residuals (residual) and actual R²
    const points: DataPoint[] = [];
    let ssTot = 0;
    let ssRes = 0;

    for (let i = 0; i < n; i++) {
      const yHatVal = m * x[i] + c;
      const residualVal = y[i] - yHatVal;
      
      ssTot += Math.pow(y[i] - meanY, 2);
      ssRes += Math.pow(residualVal, 2);

      points.push({
        index: i + 1,
        x: x[i],
        y: y[i],
        yHat: yHatVal,
        residual: residualVal,
      });
    }

    const actualR2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

    // I. Generate descriptive scatter summary text (as requested in rule 5)
    let correlationStrength = "";
    let dispersionDesc = "";

    if (target >= 0.98) {
      correlationStrength = "极强相关（微型扰动，完美契合）";
      dispersionDesc = "样本点极度紧密地贴合在回归线上，偏离均值和噪声极小，表明决定系数极其趋近于 1，因变量变化几乎完全被自变量线性解释。";
    } else if (target >= 0.8) {
      correlationStrength = "高强度正相关";
      dispersionDesc = "样本点围绕回归线呈狭窄条带状紧凑分布。数据随自变量增加而稳步上升的几何形态非常清晰，噪声信号微弱。";
    } else if (target >= 0.5) {
      correlationStrength = "中等强度正相关";
      dispersionDesc = "样本点在回归线周围呈梭形向两侧分散。虽然整体线性上升的轨迹清晰可见，但随机波动与局部离散已被明显引入。";
    } else if (target >= 0.2) {
      correlationStrength = "低强度弱相关";
      dispersionDesc = "数据表现出较为松散的漫反射形态，偏离程度很大。虽然最优拟合线仍显示正向斜率，但数据点到直线的残差平方和占比较重，噪声干扰显著。";
    } else {
      correlationStrength = "极弱相关或纯随机噪声";
      dispersionDesc = "散点完全呈无序云雾状杂乱铺开，肉眼无法直观辨识出任何上升趋势。决定系数趋向于 0，表示当前数值区间的波动基本由纯随机噪声或无法识别的外界扰动所主导。";
    }

    const sign = c >= 0 ? "+" : "-";
    const absC = Math.abs(c);

    const descriptionText = `【数据分析报告与散点分布说明】

1. 空间与数值规模约束：
   - 样本量 (N): ${n} 组二维连续实数对。
   - 自变量 X 边界: [${activeXMin.toFixed(2)}, ${activeXMax.toFixed(2)}]（区间跨度: ${(activeXMax - activeXMin).toFixed(2)}）。
   - 因变量 Y 规模: [${activeYMin.toFixed(2)}, ${activeYMax.toFixed(2)}]（完整映射，无溢出或截断）。

2. 趋势与最优回归分析：
   - 基础趋势固定为标准 一元线性正相关，回归斜率 m = ${m.toFixed(4)} > 0。
   - 经过最小二乘估计，求得的最优线性回归方程为：
     y = ${m.toFixed(4)}x ${sign} ${absC.toFixed(4)}
   - 经济/物理学释义：在当前的量级区间内，自变量 X 每递增 1 个单位，因变量 Y 的平均期望值将递增 ${m.toFixed(4)} 个单位。

3. 拟合优度与决定系数：
   - 目标设定 R² 决定系数: ${target.toFixed(4)}
   - 精准计算实际 R² 决定系数: ${actualR2.toFixed(6)}（契合度 100.00%）
   - 说明自变量 X 的变异对因变量 Y 的变异能够解释其中 ${(actualR2 * 100).toFixed(4)}% 的部分。

4. 散点分布物理状态：
   - 离散程度评级: ${correlationStrength}
   - 形态分布细则: ${dispersionDesc}`;

    return {
      points,
      m,
      c,
      actualR2,
      description: descriptionText,
      xMean: meanX,
      yMean: meanY,
      xMinActual: Math.min(...x),
      xMaxActual: Math.max(...x),
      yMinActual: Math.min(...y),
      yMaxActual: Math.max(...y),
    };
  }, [xMin, xMax, yMin, yMax, targetR2, sampleSize, seed]);

  // Copy Data Table to Clipboard
  const handleCopyTable = () => {
    const headers = "Index\tX (自变量)\tY (实际因变量)\tY_Predicted (拟合预测)\tResidual (残差)\n";
    const rows = datasetResult.points
      .map(
        (p) =>
          `${p.index}\t${p.x.toFixed(6)}\t${p.y.toFixed(6)}\t${p.yHat.toFixed(6)}\t${p.residual.toFixed(6)}`
      )
      .join("\n");
    navigator.clipboard.writeText(headers + rows).then(() => {
      triggerToast("📋 原始数据集(TSV格式)已成功复制到剪贴板！");
    });
  };

  // Copy Description Report to Clipboard
  const handleCopyDescription = () => {
    navigator.clipboard.writeText(datasetResult.description).then(() => {
      triggerToast("📋 散点分布文字说明报告已复制到剪贴板！");
    });
  };

  // Download CSV File
  const handleDownloadCSV = () => {
    const headers = "Index,X_Independent,Y_Dependent,Y_Predicted,Residual_e\n";
    const rows = datasetResult.points
      .map(
        (p) =>
          `${p.index},${p.x.toFixed(6)},${p.y.toFixed(6)},${p.yHat.toFixed(6)},${p.residual.toFixed(6)}`
      )
      .join("\n");

    const blob = new Blob([headers + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `fit_dataset_r2_${datasetResult.actualR2.toFixed(3)}_N${sampleSize}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    triggerToast("💾 CSV数据集文件已成功下载！");
  };

  // Generate random seeds to perturb noise layout while maintaining exact R2
  const handleRegenerate = () => {
    setSeed((prev) => prev + 1);
    triggerToast("🎲 随机种子已更新，生成全新扰动分布！");
  };

  return (
    <div id="main_container" className="min-h-screen bg-[#fafbfc] text-slate-800 font-sans selection:bg-indigo-500 selection:text-white">
      {/* Toast Notification */}
      {toastMessage && (
        <div id="floating_toast" className="fixed bottom-6 right-6 z-50 flex items-center gap-2 bg-slate-900 border border-slate-800 text-white px-4 py-3 rounded-xl shadow-2xl animate-fade-in font-sans text-sm">
          <Check className="w-4 h-4 text-emerald-400" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Modern Top Header Banner */}
      <header id="app_header" className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-slate-200/80 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-emerald-500 rounded-xl text-white shadow-md">
              <TrendingUp className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl md:text-2xl font-bold font-display tracking-tight text-slate-900">
                专业数据分析数据生成器
              </h1>
              <p className="text-xs md:text-sm text-slate-500 mt-0.5">
                基于 Gram-Schmidt 算法，严格配准自变量区间、因变量数值规模与 R² 决定系数
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              数学微积分引擎就绪
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
              精度: IEEE-754 双精度
            </span>
          </div>
        </div>
      </header>

      <main id="app_main" className="max-w-7xl mx-auto px-4 md:px-6 py-8">
        {/* Warning / Validation Banners */}
        {validationError && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            id="validation_alert"
            className="mb-6 p-4 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl flex items-start gap-3"
          >
            <AlertCircle className="w-5 h-5 shrink-0 text-rose-600 mt-0.5" />
            <div className="text-sm">
              <p className="font-semibold">参数输入有误</p>
              <p>{validationError}</p>
            </div>
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* ==================== LEFT PANEL: PARAMETERS ==================== */}
          <section id="params_sidebar" className="lg:col-span-4 flex flex-col gap-6">
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
              <div className="flex items-center gap-2 border-b border-slate-100 pb-4 mb-5">
                <Sliders className="w-5 h-5 text-indigo-600" />
                <h2 className="text-base font-bold font-display text-slate-900">核心约束参数设定</h2>
              </div>

              {/* Parameter 1: Independent Variable X Range */}
              <div id="param_x_range" className="space-y-3 mb-6">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-slate-700 flex items-center gap-1">
                    自变量线性区间 [X]
                    <span className="text-xs font-normal text-slate-400 font-mono">({xMin} ~ {xMax})</span>
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="text-[10px] text-slate-400 block mb-1 font-mono uppercase">X_Min (最小值)</span>
                    <input
                      type="number"
                      value={xMin}
                      onChange={(e) => setXMin(parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 hover:border-slate-300 focus:border-indigo-500 focus:bg-white outline-none rounded-lg py-2 px-3 text-sm font-mono text-slate-800 transition"
                    />
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block mb-1 font-mono uppercase">X_Max (最大值)</span>
                    <input
                      type="number"
                      value={xMax}
                      onChange={(e) => setXMax(parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 hover:border-slate-300 focus:border-indigo-500 focus:bg-white outline-none rounded-lg py-2 px-3 text-sm font-mono text-slate-800 transition"
                    />
                  </div>
                </div>
                <p className="text-[11px] text-slate-400 italic">所有生成的自变量 x 严格落在此区间内，不溢出</p>
              </div>

              {/* Parameter 2: Dependent Variable Y Scale */}
              <div id="param_y_scale" className="space-y-3 mb-6">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-slate-700 flex items-center gap-1">
                    数值整体量级规模 [Y]
                    <span className="text-xs font-normal text-slate-400 font-mono">({yMin} ~ {yMax})</span>
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="text-[10px] text-slate-400 block mb-1 font-mono uppercase">Y_Min (数值下界)</span>
                    <input
                      type="number"
                      value={yMin}
                      onChange={(e) => setYMin(parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 hover:border-slate-300 focus:border-indigo-500 focus:bg-white outline-none rounded-lg py-2 px-3 text-sm font-mono text-slate-800 transition"
                    />
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block mb-1 font-mono uppercase">Y_Max (数值上界)</span>
                    <input
                      type="number"
                      value={yMax}
                      onChange={(e) => setYMax(parseFloat(e.target.value) || 0)}
                      className="w-full bg-slate-50 border border-slate-200 hover:border-slate-300 focus:border-indigo-500 focus:bg-white outline-none rounded-lg py-2 px-3 text-sm font-mono text-slate-800 transition"
                    />
                  </div>
                </div>
                <p className="text-[11px] text-slate-400 italic">生成的数据集最大和最小 Y 严格等于指定的上下边界</p>
              </div>

              {/* Parameter 3: Target R2 */}
              <div id="param_target_r2" className="space-y-2 mb-6">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-slate-700 flex items-center gap-1">
                    目标 R² 决定系数
                  </label>
                  <span className="text-sm font-bold font-mono text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
                    {targetR2.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={targetR2}
                  onChange={(e) => setTargetR2(parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600 focus:outline-none"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-400">
                  <span>0.00 (纯白噪声)</span>
                  <span>0.50</span>
                  <span>1.00 (无损完美拟合)</span>
                </div>
                <p className="text-[11px] text-slate-400 italic">精确控制离散噪声。当 R² → 1 时紧贴直线，R² → 0 时完全无序扩散。</p>
              </div>

              {/* Extra helper: Sample size N */}
              <div id="param_sample_size" className="space-y-2 mb-6">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-slate-700">
                    样本数据量 (N)
                  </label>
                  <span className="text-sm font-bold font-mono text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded">
                    {sampleSize} 组
                  </span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="200"
                  step="5"
                  value={sampleSize}
                  onChange={(e) => setSampleSize(parseInt(e.target.value) || 10)}
                  className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-emerald-500 focus:outline-none"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-400">
                  <span>10 个点</span>
                  <span>100 个点</span>
                  <span>200 个点 (极限性能)</span>
                </div>
              </div>

              {/* Seed controller & Refresh */}
              <div className="pt-4 border-t border-slate-100 flex flex-col gap-3">
                <button
                  type="button"
                  onClick={handleRegenerate}
                  className="w-full flex items-center justify-center gap-2 bg-slate-900 hover:bg-slate-800 text-white font-medium text-sm py-2.5 px-4 rounded-xl shadow-sm hover:shadow transition-all cursor-pointer"
                >
                  <RotateCcw className="w-4 h-4 text-slate-400" />
                  生成新随机扰动 (更新种子)
                </button>
                <div className="text-[10px] text-center font-mono text-slate-400">
                  当前微积分迭代随机种子: #{seed}
                </div>
              </div>
            </div>

            {/* Model Presets Quick Select */}
            <div className="bg-white border border-slate-200 rounded-2xl p-6">
              <h3 className="text-sm font-bold font-display text-slate-900 mb-4 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-emerald-500" />
                推荐分析模型预设
              </h3>
              <div className="flex flex-col gap-2">
                <button
                  type="button"
                  onClick={() => applyPreset("perfect")}
                  className="flex items-center justify-between text-left px-3 py-2 text-xs rounded-xl hover:bg-slate-50 border border-slate-100 transition"
                >
                  <span className="font-semibold text-slate-700">💎 极优物理常量模型</span>
                  <span className="font-mono text-slate-400">R² = 1.00</span>
                </button>
                <button
                  type="button"
                  onClick={() => applyPreset("strong")}
                  className="flex items-center justify-between text-left px-3 py-2 text-xs rounded-xl hover:bg-slate-50 border border-slate-100 transition"
                >
                  <span className="font-semibold text-slate-700">📈 强正向金融增长趋势</span>
                  <span className="font-mono text-slate-400">R² = 0.92</span>
                </button>
                <button
                  type="button"
                  onClick={() => applyPreset("moderate")}
                  className="flex items-center justify-between text-left px-3 py-2 text-xs rounded-xl hover:bg-slate-50 border border-slate-100 transition"
                >
                  <span className="font-semibold text-slate-700">📊 自然界经典生长关联</span>
                  <span className="font-mono text-slate-400">R² = 0.65</span>
                </button>
                <button
                  type="button"
                  onClick={() => applyPreset("weak")}
                  className="flex items-center justify-between text-left px-3 py-2 text-xs rounded-xl hover:bg-slate-50 border border-slate-100 transition"
                >
                  <span className="font-semibold text-slate-700">🌪️ 高残差气象干扰模型</span>
                  <span className="font-mono text-slate-400">R² = 0.25</span>
                </button>
                <button
                  type="button"
                  onClick={() => applyPreset("zero")}
                  className="flex items-center justify-between text-left px-3 py-2 text-xs rounded-xl hover:bg-slate-50 border border-slate-100 transition"
                >
                  <span className="font-semibold text-slate-700">🎲 纯随机布朗运动白噪声</span>
                  <span className="font-mono text-slate-400">R² = 0.00</span>
                </button>
              </div>
            </div>
          </section>

          {/* ==================== RIGHT PANEL: VISUALIZATION & OUTPUTS ==================== */}
          <section id="analytics_content" className="lg:col-span-8 flex flex-col gap-6">
            
            {/* 1. Core KPIs summary cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              
              {/* Regression equation card */}
              <motion.div
                key={`m-c-${datasetResult.m}-${datasetResult.c}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm"
              >
                <span className="text-[10px] font-bold text-slate-400 block tracking-wider uppercase mb-1">
                  最优线性回归方程
                </span>
                <p className="text-lg font-bold font-mono text-indigo-600 truncate">
                  y = {datasetResult.m.toFixed(4)}x {datasetResult.c >= 0 ? "+" : "-"} {Math.abs(datasetResult.c).toFixed(4)}
                </p>
                <div className="text-[11px] text-slate-400 mt-2">
                  斜率 (Slope) m = {datasetResult.m.toFixed(4)}，截距 (Intercept) c = {datasetResult.c.toFixed(4)}
                </div>
              </motion.div>

              {/* R2 verification card */}
              <motion.div
                key={`r2-${datasetResult.actualR2}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-indigo-950 text-white p-5 rounded-2xl shadow-sm"
              >
                <span className="text-[10px] font-bold text-indigo-300 block tracking-wider uppercase mb-1">
                  精准计算实际决定系数 (R²)
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-3xl font-extrabold font-mono text-emerald-400">
                    {datasetResult.actualR2.toFixed(6)}
                  </span>
                  <span className="text-xs text-indigo-300 font-mono">
                    (100.00% 匹配)
                  </span>
                </div>
                <div className="text-[11px] text-indigo-200 mt-2">
                  目标 R² 设定为: <span className="font-bold">{targetR2.toFixed(2)}</span>
                </div>
              </motion.div>

              {/* Quality rating card */}
              <motion.div
                key={`desc-level-${datasetResult.actualR2}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm flex flex-col justify-between"
              >
                <div>
                  <span className="text-[10px] font-bold text-slate-400 block tracking-wider uppercase mb-1">
                    趋势离散评级
                  </span>
                  <span className={`inline-flex px-2 py-0.5 rounded text-xs font-semibold ${
                    datasetResult.actualR2 >= 0.9 ? "bg-emerald-50 text-emerald-700 border border-emerald-100" :
                    datasetResult.actualR2 >= 0.6 ? "bg-indigo-50 text-indigo-700 border border-indigo-100" :
                    datasetResult.actualR2 >= 0.3 ? "bg-amber-50 text-amber-700 border border-amber-100" :
                    "bg-rose-50 text-rose-700 border border-rose-100"
                  }`}>
                    {datasetResult.actualR2 >= 0.95 ? "极优拟合 (Perfect Fit)" :
                     datasetResult.actualR2 >= 0.8 ? "强正相关 (Strong Correlation)" :
                     datasetResult.actualR2 >= 0.5 ? "中度离散 (Moderate Noise)" :
                     datasetResult.actualR2 >= 0.15 ? "高残差离散 (High Residuals)" :
                     "纯白噪声/无显著关联"}
                  </span>
                </div>
                <div className="text-[11px] text-slate-400 mt-2">
                  自变量对因变量波动的线性解释度为 <span className="font-semibold text-slate-700">{(datasetResult.actualR2 * 100).toFixed(2)}%</span>。
                </div>
              </motion.div>
            </div>

            {/* 2. Interactive Scatter & Line Plot */}
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
                <div>
                  <h3 className="text-base font-bold font-display text-slate-900">
                    拟合曲线与散点态势图
                  </h3>
                  <p className="text-xs text-slate-400">
                    散点为样本观测点，深蓝色直线为回归方程 y = mx + c 的数学投影
                  </p>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <div className="flex items-center gap-1.5 text-slate-500">
                    <span className="w-2.5 h-2.5 rounded-full bg-indigo-600 block"></span>
                    <span>观测点 (Actual)</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-slate-500">
                    <span className="w-5 h-0.5 bg-blue-500 block"></span>
                    <span>最佳回归线 (Regression)</span>
                  </div>
                </div>
              </div>

              {/* Chart container */}
              <div id="scatter_chart" className="w-full h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={datasetResult.points}
                    margin={{ top: 20, right: 10, bottom: 5, left: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis
                      dataKey="x"
                      type="number"
                      domain={[xMin, xMax]}
                      stroke="#94a3b8"
                      fontSize={11}
                      tickLine={false}
                      tickFormatter={(v) => v.toFixed(1)}
                    />
                    <YAxis
                      dataKey="y"
                      type="number"
                      domain={[yMin, yMax]}
                      stroke="#94a3b8"
                      fontSize={11}
                      tickLine={false}
                      tickFormatter={(v) => v.toFixed(1)}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Scatter
                      name="真实观测值"
                      dataKey="y"
                      fill="#4f46e5"
                      fillOpacity={0.85}
                      stroke="#ffffff"
                      strokeWidth={1}
                    />
                    <Line
                      name="回归方程线"
                      dataKey="yHat"
                      stroke="#3b82f6"
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Range Verification footer */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-100 text-xs font-mono text-slate-500">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase">X 实际跨度</span>
                  <span>{datasetResult.xMinActual.toFixed(2)} ~ {datasetResult.xMaxActual.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase">Y 实际极值</span>
                  <span>{datasetResult.yMinActual.toFixed(2)} ~ {datasetResult.yMaxActual.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase">自变量均值 (X̄)</span>
                  <span>{datasetResult.xMean.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase">因变量均值 (Ȳ)</span>
                  <span>{datasetResult.yMean.toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* 3. Generated Statistical Description Summary (Rule 5 requirement) */}
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-4">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-indigo-600" />
                  <h3 className="text-base font-bold font-display text-slate-900">
                    简易散点分布文字说明
                  </h3>
                </div>
                <button
                  type="button"
                  onClick={handleCopyDescription}
                  className="flex items-center gap-1 bg-slate-50 hover:bg-slate-100 text-slate-600 font-medium text-xs px-3 py-1.5 rounded-lg border border-slate-200 transition-all cursor-pointer"
                >
                  <Copy className="w-3.5 h-3.5" />
                  复制说明报告
                </button>
              </div>

              {/* Formatted Text Box */}
              <div id="description_box" className="bg-slate-50 border border-slate-100 p-5 rounded-xl text-xs md:text-sm text-slate-700 font-mono leading-relaxed whitespace-pre-wrap select-text">
                {datasetResult.description}
              </div>
            </div>

            {/* 4. Tabular Data Inspector & Exporter (Rule 5 requirement) */}
            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
              <div className="p-6 border-b border-slate-200/80 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                  <Database className="w-5 h-5 text-emerald-500" />
                  <div>
                    <h3 className="text-base font-bold font-display text-slate-900">
                      原始成对 XY 数据集
                    </h3>
                    <p className="text-xs text-slate-400">
                      展示前 100 组实数坐标点，包含最小二乘回归预测值与残差计算
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleCopyTable}
                    className="flex items-center gap-1 bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium text-xs px-3 py-2 rounded-xl border border-slate-200 transition-all cursor-pointer"
                  >
                    <Copy className="w-3.5 h-3.5 text-slate-500" />
                    复制制表符数据 (Clipboard)
                  </button>
                  <button
                    type="button"
                    onClick={handleDownloadCSV}
                    className="flex items-center gap-1.5 bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-xs px-3 py-2 rounded-xl shadow-sm transition-all cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    下载完整 CSV
                  </button>
                </div>
              </div>

              {/* Data Table Wrapper */}
              <div id="data_table_wrapper" className="max-h-[300px] overflow-y-auto overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 sticky top-0 z-10 text-xs font-semibold text-slate-600 font-display">
                      <th className="px-4 py-3 font-mono">序号 (Index)</th>
                      <th className="px-4 py-3">自变量 (X)</th>
                      <th className="px-4 py-3">因变量真实值 (Y)</th>
                      <th className="px-4 py-3">回归预测值 (Ŷ)</th>
                      <th className="px-4 py-3">残差误差 (e)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-xs font-mono text-slate-700">
                    {datasetResult.points.slice(0, 100).map((point) => (
                      <tr key={point.index} className="hover:bg-indigo-50/45 transition">
                        <td className="px-4 py-2.5 text-slate-400">#{point.index}</td>
                        <td className="px-4 py-2.5 font-semibold text-slate-800">
                          {point.x.toFixed(6)}
                        </td>
                        <td className="px-4 py-2.5 text-indigo-600 font-semibold">
                          {point.y.toFixed(6)}
                        </td>
                        <td className="px-4 py-2.5 text-slate-600">
                          {point.yHat.toFixed(6)}
                        </td>
                        <td className="px-4 py-2.5">
                          <span className={point.residual >= 0 ? "text-emerald-600" : "text-rose-600"}>
                            {point.residual >= 0 ? "+" : ""}
                            {point.residual.toFixed(6)}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {datasetResult.points.length > 100 && (
                      <tr>
                        <td colSpan={5} className="px-4 py-3 text-center bg-slate-50 text-slate-400 font-sans italic text-xs">
                          已省略其余 {datasetResult.points.length - 100} 条数据。请点击右上方按钮下载完整 CSV。
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

          </section>
        </div>
      </main>

      {/* Modern statistical guide footer */}
      <footer id="app_footer" className="max-w-7xl mx-auto px-6 py-12 mt-12 border-t border-slate-200">
        <div className="flex flex-col md:flex-row items-start justify-between gap-8">
          <div className="space-y-2 max-w-lg">
            <h4 className="text-sm font-bold font-display text-slate-900 flex items-center gap-1">
              <Info className="w-4 h-4 text-indigo-600" />
              Gram-Schmidt 回归配准算法原理
            </h4>
            <p className="text-xs text-slate-500 leading-relaxed">
              为了达到决定系数 R² 的精确值，本生成器先在指定区间生成随机特征分布并对其标准化。随后引入高斯白噪声，通过施密特正交化投影消去噪声在特征上的线性相关分量，得到与自变量正交（相关系数为0）的纯残差，再利用几何乘子 √R² 与 √(1-R²) 线性权重合成目标输出，最后运用严格保正相关性的映射函数归一化到所需规模中。
            </p>
          </div>
          <div className="text-xs text-slate-400 md:text-right">
            <p>© 2026 线性拟合数据生成器. All Rights Reserved.</p>
            <p className="mt-1 font-mono">Precision: Real64 Math Engine v2.0</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Tooltip rendering for Recharts ComposedChart
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-slate-900 border border-slate-800 p-4.5 rounded-xl shadow-2xl text-[11px] font-sans text-white max-w-[220px]">
        <p className="font-semibold text-slate-400 mb-2 border-b border-slate-800 pb-1 flex items-center justify-between">
          <span>样本点 #{data.index}</span>
          <span className="text-[9px] bg-indigo-500/20 text-indigo-300 px-1 py-0.5 rounded">
            Observed
          </span>
        </p>
        <div className="space-y-1.5 font-mono">
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">自变量 X:</span>
            <span className="text-emerald-400 font-bold">{data.x.toFixed(4)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">因变量 Y:</span>
            <span className="text-indigo-300 font-bold">{data.y.toFixed(4)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className="text-slate-400">回归预测 Ŷ:</span>
            <span className="text-blue-400 font-bold">{data.yHat.toFixed(4)}</span>
          </div>
          <div className="flex justify-between gap-4 border-t border-slate-800 pt-1">
            <span className="text-slate-400">残差 (e):</span>
            <span className={data.residual >= 0 ? "text-emerald-400" : "text-rose-400"}>
              {(data.residual >= 0 ? "+" : "") + data.residual.toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};
