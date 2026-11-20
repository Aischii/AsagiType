//! Background thread that owns the candle models (loaded on first use) and runs
//! detection + OCR and cleaning on page images.

use eframe::egui;
use std::{
    io::Cursor,
    path::PathBuf,
    sync::mpsc::{channel, Receiver, Sender},
    thread,
};
use vision::{
    clean::{clean_page, CleanBox},
    Models, TextRegion,
};

pub enum Request {
    Analyze {
        page: usize,
        bytes: Vec<u8>,
        replace: bool,
    },
    Clean {
        page: usize,
        bytes: Vec<u8>,
        boxes: Vec<CleanBox>,
    },
}

pub enum Outcome {
    Analyzed {
        replace: bool,
        regions: Vec<TextRegion>,
    },
    /// PNG-encoded RGBA clean layer.
    Cleaned { png: Vec<u8> },
}

pub struct PageResult {
    pub page: usize,
    pub result: Result<Outcome, String>,
}

pub struct VisionWorker {
    tx: Sender<Request>,
    rx: Receiver<PageResult>,
    pub models_dir: PathBuf,
    pub pending: usize,
}

const THRESHOLD: f32 = 0.5;

fn run(
    models: &mut Option<Models>,
    dir: &std::path::Path,
    req: Request,
) -> (usize, Result<Outcome, String>) {
    match req {
        Request::Analyze {
            page,
            bytes,
            replace,
        } => {
            let result = (|| {
                if models.is_none() {
                    *models = Some(Models::load(dir).map_err(|e| {
                        format!("Could not load models from {}:\n{e:#}", dir.display())
                    })?);
                }
                let img = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
                let regions = models
                    .as_ref()
                    .unwrap()
                    .analyze(&img, THRESHOLD)
                    .map_err(|e| format!("{e:#}"))?;
                Ok(Outcome::Analyzed { replace, regions })
            })();
            (page, result)
        }
        Request::Clean { page, bytes, boxes } => {
            let result = (|| {
                let img = image::load_from_memory(&bytes)
                    .map_err(|e| e.to_string())?
                    .to_rgb8();
                let layer = clean_page(&img, &boxes);
                let mut png = Vec::new();
                layer
                    .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
                    .map_err(|e| e.to_string())?;
                Ok(Outcome::Cleaned { png })
            })();
            (page, result)
        }
    }
}

impl VisionWorker {
    pub fn spawn(models_dir: PathBuf, ctx: egui::Context) -> Self {
        let (tx, req_rx) = channel::<Request>();
        let (res_tx, rx) = channel::<PageResult>();
        let dir = models_dir.clone();
        thread::Builder::new()
            .name("asagitype-vision".into())
            .spawn(move || {
                let mut models: Option<Models> = None;
                for req in req_rx {
                    let (page, result) = run(&mut models, &dir, req);
                    let _ = res_tx.send(PageResult { page, result });
                    ctx.request_repaint();
                }
            })
            .expect("spawn vision thread");
        Self {
            tx,
            rx,
            models_dir,
            pending: 0,
        }
    }

    pub fn submit(&mut self, req: Request) {
        if self.tx.send(req).is_ok() {
            self.pending += 1;
        }
    }

    pub fn poll(&mut self) -> Vec<PageResult> {
        let out: Vec<PageResult> = self.rx.try_iter().collect();
        self.pending = self.pending.saturating_sub(out.len());
        out
    }
}
