use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};

// ==================== CDF (Abramowitz & Stegun) ====================
/// Стандарт хэвийн тархалтын CDF, нарийвчлал ~7.5e-8
fn normal_cdf(z: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989423 * (-z * z / 2.0).exp();
    let p = d
        * t
        * (0.3193815
            + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if z > 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// P(X > level) — median, q10, q90-аас sigma тооцоолно
fn prob_above(median: f64, q10: f64, q90: f64, level: f64) -> f64 {
    let sigma = (q90 - q10) / (2.0 * 1.2816);
    if sigma <= 0.0 {
        return 50.0;
    }
    let z = (level - median) / sigma;
    (1.0 - normal_cdf(z)) * 100.0
}

/// P(UP) = P(Close > prev_close)
fn prob_up(prev_close: f64, median: f64, q10: f64, q90: f64) -> f64 {
    prob_above(median, q10, q90, prev_close)
}

// ==================== JSON бүтэц ====================
#[derive(Debug, Deserialize)]
struct InputData {
    context: Vec<Vec<Vec<f64>>>,
    horizon: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ForecastOutput {
    point: Vec<f64>,
    quantiles: BTreeMap<String, Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct ModelResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    forecast: Forecast,
}

#[derive(Debug, Deserialize)]
struct Forecast {
    variates: Vec<ForecastOutput>,
}

// ==================== Загвар дуудах ====================
fn run_model(model: &str, input_json: &str) -> Result<Vec<ForecastOutput>, String> {
    let (program, args): (&str, Vec<&str>) = match model {
        "chronos" => (
            "../target/release/zsfm",
            vec![
                "chronos", "infer",
                "--gguf", "../gguf/chronos-f16.gguf",
                "--config", "../models/amazon__chronos-2/config.json",
                "--group-ids", "0,0,0,0,0",
            ],
        ),
        "toto" => (
            "../target/release/zsfm",
            vec![
                "toto", "infer",
                "--gguf", "../gguf/toto-22m-q8.gguf",
                "--config", "../models/Datadog__Toto-2.0-22m/config.json",
            ],
        ),
        _ => return Err(format!("Unknown model: {}", model)),
    };

    let mut child = Command::new(program)
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("spawn failed: {}", e))?;

    {
        let stdin = child.stdin.as_mut().ok_or("stdin open failed")?;
        stdin
            .write_all(input_json.as_bytes())
            .map_err(|e| format!("write failed: {}", e))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("wait failed: {}", e))?;
    if !output.status.success() {
        return Err(format!("{} failed: {:?}", model, output.status));
    }

    let resp: ModelResponse = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("parse JSON failed: {}", e))?;

    Ok(resp.choices[0].forecast.variates.clone())
}

// ==================== Үндсэн програм ====================
fn main() {
    // Оролт унших
    let input_str = match fs::read_to_string("../gold_ohlcv.json") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gold_ohlcv.json уншиж чадсангүй: {}", e);
            std::process::exit(1);
        }
    };
    let input: InputData = match serde_json::from_str(&input_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("JSON parse error: {}", e);
            std::process::exit(1);
        }
    };

    let last_close = input.context[0][3][input.context[0][3].len() - 1];
    let n_hor = input.horizon;

    println!();
    println!("Одоогийн Close: ${:.2}", last_close);
    println!(
        "Context: {} өдөр, Horizon: {} өдөр",
        input.context[0][0].len(),
        n_hor
    );

    // Chronos-2
    eprintln!("Running Chronos-2...");
    let ch_variates = match run_model("chronos", &input_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Chronos error: {}", e);
            std::process::exit(1);
        }
    };

    // Toto-2 22M
    eprintln!("Running Toto-2 22M...");
    let tt_variates = match run_model("toto", &input_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Toto error: {}", e);
            std::process::exit(1);
        }
    };

    // Close variate (index 3)
    let ch_close = &ch_variates[3];
    let tt_close = &tt_variates[3];
    let ch_med = &ch_close.point;
    let tt_med = &tt_close.point;
    let ch_q = &ch_close.quantiles;
    let tt_q = &tt_close.quantiles;

    // ==================== UP/DOWN МАГАДЛАЛ ====================
    println!();
    println!("{}", "=".repeat(110));
    println!("=== МАГАДЛАЛЫН ШИНЖИЛГЭЭ — XAUUSD ({} өдөр) ===", n_hor);
    println!("=== Суурь: Close = ${:.2} ===", last_close);
    println!("{}", "=".repeat(110));

    println!();
    println!("1. UP/DOWN МАГАДЛАЛ (өмнөх өдрийн Close-той харьцуулан)");
    println!("{}", "-".repeat(110));
    println!(
        "{:<6} {:>13} {:>13} {:>12} {:>12} {:>12}",
        "Day", "Chronos Med", "Chronos UP%", "Toto Med", "Toto UP%", "Consensus"
    );
    println!("{}", "-".repeat(110));

    let mut prev_ch = last_close;
    let mut prev_tt = last_close;
    for i in 0..n_hor {
        let ch_p = prob_up(
            prev_ch,
            ch_med[i],
            ch_q["0.10"][i],
            ch_q["0.90"][i],
        );
        let tt_p = prob_up(
            prev_tt,
            tt_med[i],
            tt_q["0.10"][i],
            tt_q["0.90"][i],
        );
        let consensus = (ch_p + tt_p) / 2.0;
        let marker = if consensus > 60.0 {
            " ↑↑"
        } else if consensus > 55.0 {
            " ↑"
        } else if consensus < 40.0 {
            " ↓↓"
        } else if consensus < 45.0 {
            " ↓"
        } else {
            ""
        };
        println!(
            "D+{:<3} {:>13.2} {:>12.1}% {:>12.2} {:>11.1}% {:>11.1}%{}",
            i + 1,
            ch_med[i],
            ch_p,
            tt_med[i],
            tt_p,
            consensus,
            marker
        );
        prev_ch = ch_med[i];
        prev_tt = tt_med[i];
    }

    // ==================== 80% ИНТГЭЛТЭЙ ИНТЕРВАЛ ====================
    println!();
    println!("2. ИТГЭЛТЭЙ ИНТЕРВАЛУУД (80%)");
    println!("{}", "-".repeat(110));
    println!("{:<6} {:<32} {:<32}", "Day", "Chronos 80%", "Toto 80%");
    println!("{}", "-".repeat(110));
    for i in 0..n_hor {
        let ch_lo = ch_q["0.10"][i];
        let ch_hi = ch_q["0.90"][i];
        let tt_lo = tt_q["0.10"][i];
        let tt_hi = tt_q["0.90"][i];
        println!(
            "D+{:<3} [{:8.2} .. {:8.2}]  [{:8.2} .. {:8.2}]",
            i + 1,
            ch_lo,
            ch_hi,
            tt_lo,
            tt_hi
        );
    }

    // ==================== ТҮВШИН ХҮРЭХ МАГАДЛАЛ ====================
    println!();
    println!("3. ТҮВШИН ХҮРЭХ МАГАДЛАЛ (Chronos-2, D+{})", n_hor);
    println!("{}", "-".repeat(110));

    let d_last = n_hor - 1;
    let ch_med_last = ch_med[d_last];
    let q10_last = ch_q["0.10"][d_last];
    let q90_last = ch_q["0.90"][d_last];

    println!("Суурь Close: ${:.2}", last_close);
    println!("D+{} Median: ${:.2}", n_hor, ch_med_last);
    println!();
    println!(
        "{:<12} {:>20} {:>20}",
        "Level", "P(Close > Level)", "P(Close < Level)"
    );
    println!("{}", "-".repeat(60));

    let levels = [
        last_close - 200.0,
        last_close - 100.0,
        last_close - 50.0,
        last_close,
        last_close + 50.0,
        last_close + 100.0,
        last_close + 200.0,
        ch_med_last,
    ];

    for &lvl in &levels {
        let p_above = prob_above(ch_med_last, q10_last, q90_last, lvl);
        let p_below = 100.0 - p_above;
        let marker = if (lvl - last_close).abs() < 0.01 {
            " ← одоогийн"
        } else if (lvl - ch_med_last).abs() < 0.01 {
            " ← median"
        } else {
            ""
        };
        println!(
            "${:<11.2} {:>19.1}% {:>19.1}%{}",
            lvl, p_above, p_below, marker
        );
    }

    // ==================== ХҮЛЭЭГДЭЖ БУЙ ХӨДӨЛГӨӨН ====================
    println!();
    println!("4. ХҮЛЭЭГДЭЖ БУЙ ХӨДӨЛГӨӨН (D+{})", n_hor);
    println!("{}", "-".repeat(110));
    let ch_move = ch_med_last - last_close;
    let tt_move = tt_med[d_last] - last_close;
    println!(
        "{:<12} {:>15} {:>15} {:>15}",
        "Model", "D+10 Median", "Move ($)", "Move (%)"
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:<12} {:>15.2} {:>+15.2} {:>+14.2}%",
        "Chronos-2",
        ch_med_last,
        ch_move,
        ch_move / last_close * 100.0
    );
    println!(
        "{:<12} {:>15.2} {:>+15.2} {:>+14.2}%",
        "Toto 22M",
        tt_med[d_last],
        tt_move,
        tt_move / last_close * 100.0
    );

    // ==================== ХУРИМТЛАЛТ МАГАДЛАЛ ====================
    println!();
    println!("5. ХУРИМТЛАЛТ МАГАДЛАЛ (Chronos-2, D+{})", n_hor);
    println!("{}", "-".repeat(110));
    println!("P(Close_D+{} < X):", n_hor);
    for &lvl in &levels {
        let p = prob_above(ch_med_last, q10_last, q90_last, lvl);
        let p_below = 100.0 - p;
        let bars = (p_below / 5.0) as usize;
        let bar_str: String = "█".repeat(bars);
        println!("  < ${:8.2}: {:5.1}%  {}", lvl, p_below, bar_str);
    }

    // ==================== ЭРСДЭЛИЙН ХЭМЖЭЭ ====================
    println!();
    println!("6. ЭРСДЭЛИЙН ХЭМЖЭЭ");
    println!("{}", "-".repeat(110));
    println!("{:<30} {:>15} {:>15}", "Metric", "Chronos-2", "Toto 22M");
    println!("{}", "-".repeat(60));
    println!(
        "{:<30} {:>15.2} {:>15.2}",
        "D+10 Median ($)", ch_med_last, tt_med[d_last]
    );
    println!(
        "{:<30} {:>15.2} {:>15.2}",
        "D+10 q0.10 ($)", q10_last, tt_q["0.10"][d_last]
    );
    println!(
        "{:<30} {:>15.2} {:>15.2}",
        "D+10 q0.90 ($)", q90_last, tt_q["0.90"][d_last]
    );
    println!(
        "{:<30} {:>15.2} {:>15.2}",
        "80% width ($)",
        q90_last - q10_last,
        tt_q["0.90"][d_last] - tt_q["0.10"][d_last]
    );
    println!(
        "{:<30} {:>14.2}% {:>14.2}%",
        "80% width (%)",
        (q90_last - q10_last) / last_close * 100.0,
        (tt_q["0.90"][d_last] - tt_q["0.10"][d_last]) / last_close * 100.0
    );

    println!();
    println!("{}", "=".repeat(110));
    println!("=== ДУУССАН ===");
    println!("{}", "=".repeat(110));
}
