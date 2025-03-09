use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use terminal_size::{terminal_size, Width, Height};
use once_cell::sync::Lazy;
use chrono::Local;

// A global tracker that can be accessed from anywhere
pub static PROGRESS_TRACKER: Lazy<Arc<Mutex<ProgressTracker>>> = Lazy::new(|| {
    Arc::new(Mutex::new(ProgressTracker::new()))
});

// Log levels
#[derive(Clone, Copy)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
    Debug,
}

// Represents the different stages of processing
#[derive(Clone, Copy, PartialEq)]
pub enum ProcessingStage {
    Global,
    ConfigEntry,
    VcfProcessing,
    VariantAnalysis,
    CdsProcessing,
    StatsCalculation,
}

// Represents a status box with title and key-value pairs
#[derive(Clone)]
pub struct StatusBox {
    pub title: String,
    pub stats: Vec<(String, String)>,
}

pub struct ProgressTracker {
    // The multi-progress manages all progress bars
    multi_progress: MultiProgress,
    
    // Main progress indicators for different stages
    global_bar: Option<ProgressBar>,
    entry_bar: Option<ProgressBar>,
    step_bar: Option<ProgressBar>,
    variant_bar: Option<ProgressBar>,
    
    // Track the current stage and state
    current_stage: ProcessingStage,
    total_entries: usize,
    current_entry: usize,
    entry_name: String,
    
    // Log file writers
    processing_log: Option<BufWriter<File>>,
    variants_log: Option<BufWriter<File>>,
    transcripts_log: Option<BufWriter<File>>,
    stats_log: Option<BufWriter<File>>,
    
    // Track timing for operations
    start_time: Instant,
    
    // Directory for log files
    log_dir: PathBuf,
    
    // Cache for reusable progress styles
    styles: HashMap<String, ProgressStyle>,
}

impl ProgressTracker {
    pub fn new() -> Self {
        let multi_progress = MultiProgress::new();
        
        // Create default log directory
        let log_dir = PathBuf::from("ferromic_logs");
        let _ = fs::create_dir_all(&log_dir);
        
        // Create reusable styles
        let mut styles = HashMap::new();
        
        // Global progress style
        styles.insert(
            "global".to_string(),
            ProgressStyle::default_bar()
                .template("{spinner:.blue} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} entries {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░")
        );
        
        // Entry progress style
        styles.insert(
            "entry".to_string(),
            ProgressStyle::default_bar()
                .template("  {spinner:.green} [{elapsed_precise}] {bar:30.green/white} {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░")
        );
        
        // Step progress style
        styles.insert(
            "step".to_string(),
            ProgressStyle::default_bar()
                .template("    {spinner:.yellow} [{elapsed_precise}] {bar:20.yellow/white} {pos}/{len} {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░")
        );
        
        // Variant progress style
        styles.insert(
            "variant".to_string(),
            ProgressStyle::default_bar()
                .template("      {spinner:.magenta} [{elapsed_precise}] {bar:15.magenta/white} {pos}/{len} {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░")
        );
        
        // Spinner style
        styles.insert(
            "spinner".to_string(),
            ProgressStyle::default_spinner()
                .template("{spinner:.bold.green} {msg} {elapsed_precise}")
                .expect("Spinner template error")
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        );
        
        ProgressTracker {
            multi_progress,
            global_bar: None,
            entry_bar: None,
            step_bar: None,
            variant_bar: None,
            current_stage: ProcessingStage::Global,
            total_entries: 0,
            current_entry: 0,
            entry_name: String::new(),
            processing_log: Self::create_log_file(&log_dir, "processing.log"),
            variants_log: Self::create_log_file(&log_dir, "variants.log"),
            transcripts_log: Self::create_log_file(&log_dir, "transcripts.log"),
            stats_log: Self::create_log_file(&log_dir, "stats.log"),
            start_time: Instant::now(),
            log_dir,
            styles,
        }
    }
    
    fn create_log_file(dir: &Path, filename: &str) -> Option<BufWriter<File>> {
        match OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(dir.join(filename))
        {
            Ok(file) => Some(BufWriter::new(file)),
            Err(e) => {
                eprintln!("Error creating log file {}: {}", filename, e);
                None
            }
        }
    }
    
    pub fn init_global_progress(&mut self, total: usize) {
        self.total_entries = total;
        self.current_entry = 0;
        
        let style = self.styles.get("global").cloned().unwrap_or_else(|| {
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} entries {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░")
        });
        
        let bar = self.multi_progress.add(ProgressBar::new(total as u64));
        bar.set_style(style);
        bar.set_message("Processing config entries...".to_string());
        
        self.global_bar = Some(bar);
        self.log(LogLevel::Info, &format!("Starting processing of {} config entries", total));
    }
    
    pub fn update_global_progress(&mut self, current: usize, message: &str) {
        if let Some(bar) = &self.global_bar {
            self.current_entry = current;
            bar.set_position(current as u64);
            bar.set_message(message.to_string());
        }
    }
    
    pub fn init_entry_progress(&mut self, entry_desc: &str, len: u64) {
        // Clear any existing entry progress
        if let Some(old_bar) = self.entry_bar.take() {
            old_bar.finish_and_clear();
        }
        
        // Also clear step and variant bars
        if let Some(old_bar) = self.step_bar.take() {
            old_bar.finish_and_clear();
        }
        
        if let Some(old_bar) = self.variant_bar.take() {
            old_bar.finish_and_clear();
        }
        
        let style = self.styles.get("entry").cloned().unwrap_or_else(|| {
            ProgressStyle::default_bar()
                .template("  [{elapsed_precise}] {bar:30.green/white} {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░")
        });
        
        let bar = self.multi_progress.add(ProgressBar::new(len));
        bar.set_style(style);
        bar.set_message(format!("Processing {}", entry_desc));
        
        self.entry_bar = Some(bar);
        self.entry_name = entry_desc.to_string();
        self.log(LogLevel::Info, &format!("Processing entry: {}", entry_desc));
    }
    
    pub fn update_entry_progress(&mut self, position: u64, message: &str) {
        if let Some(bar) = &self.entry_bar {
            bar.set_position(position);
            bar.set_message(message.to_string());
        }
    }
    
    pub fn finish_entry_progress(&mut self, message: &str) {
        if let Some(bar) = &self.entry_bar {
            bar.finish_with_message(message.to_string());
        }
        
        // Also update global progress
        if let Some(bar) = &self.global_bar {
            self.current_entry += 1;
            bar.set_position(self.current_entry as u64);
            bar.set_message(format!("Completed {}/{}: {}", 
                self.current_entry, self.total_entries, self.entry_name));
        }
    }
    
    pub fn init_step_progress(&mut self, step_desc: &str, len: u64) {
        // Clear any existing step progress
        if let Some(old_bar) = self.step_bar.take() {
            old_bar.finish_and_clear();
        }
        
        // Also clear variant bar
        if let Some(old_bar) = self.variant_bar.take() {
            old_bar.finish_and_clear();
        }
        
        let style = self.styles.get("step").cloned().unwrap_or_else(|| {
            ProgressStyle::default_bar()
                .template("    [{elapsed_precise}] {bar:20.yellow/white} {pos}/{len} {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░")
        });
        
        let bar = self.multi_progress.add(ProgressBar::new(len));
        bar.set_style(style);
        bar.set_message(format!("{}", step_desc));
        
        self.step_bar = Some(bar);
        self.log(LogLevel::Info, &format!("Starting step: {}", step_desc));
    }
    
    pub fn update_step_progress(&mut self, position: u64, message: &str) {
        if let Some(bar) = &self.step_bar {
            bar.set_position(position);
            bar.set_message(message.to_string());
        }
    }
    
    pub fn finish_step_progress(&mut self, message: &str) {
        if let Some(bar) = &self.step_bar {
            bar.finish_with_message(message.to_string());
        }
    }
    
    pub fn init_variant_progress(&mut self, desc: &str, len: u64) {
        // Clear any existing variant progress bar
        if let Some(old_bar) = self.variant_bar.take() {
            old_bar.finish_and_clear();
        }
        
        let style = self.styles.get("variant").cloned().unwrap_or_else(|| {
            ProgressStyle::default_bar()
                .template("      [{elapsed_precise}] {bar:15.magenta/white} {pos}/{len} {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░")
        });
        
        let bar = self.multi_progress.add(ProgressBar::new(len));
        bar.set_style(style);
        bar.set_message(format!("{}", desc));
        
        self.variant_bar = Some(bar);
        self.log(LogLevel::Debug, &format!("Starting variant analysis: {}", desc));
    }
    
    pub fn update_variant_progress(&mut self, position: u64, message: &str) {
        if let Some(bar) = &self.variant_bar {
            bar.set_position(position);
            bar.set_message(message.to_string());
        }
    }
    
    pub fn finish_variant_progress(&mut self, message: &str) {
        if let Some(bar) = &self.variant_bar {
            bar.finish_with_message(message.to_string());
        }
    }
    
    pub fn spinner(&mut self, message: &str) -> ProgressBar {
        // Spinner with stage context and better timing information
        let stage_indicator = match self.current_stage {
            ProcessingStage::Global => "[Global]",
            ProcessingStage::ConfigEntry => "[Entry]",
            ProcessingStage::VcfProcessing => "[VCF]",
            ProcessingStage::VariantAnalysis => "[Variant]",
            ProcessingStage::CdsProcessing => "[CDS]",
            ProcessingStage::StatsCalculation => "[Stats]",
        };
    
        let style = self.styles.get("spinner").cloned().unwrap_or_else(|| {
            ProgressStyle::default_spinner()
                .template(&format!("{{spinner:.bold.green}} {} {{msg}} [{{elapsed_precise}}]", stage_indicator))
                .expect("Spinner template error")
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        });
    
        let spinner = self.multi_progress.add(ProgressBar::new_spinner());
        spinner.set_style(style);
        spinner.set_message(message.to_string());
        spinner.enable_steady_tick(Duration::from_millis(80));
    
        self.log(LogLevel::Info, &format!("Started operation: {}", message));
        spinner
    }
    
    pub fn log(&mut self, level: LogLevel, message: &str) {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
        let level_str = match level {
            LogLevel::Info => "INFO",
            LogLevel::Warning => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Debug => "DEBUG",
        };
        
        let log_line = format!("[{}] [{}] {}\n", timestamp, level_str, message);
        
        // Write to the appropriate log file based on current stage
        let log_file = match self.current_stage {
            ProcessingStage::VcfProcessing => &mut self.variants_log,
            ProcessingStage::CdsProcessing => &mut self.transcripts_log,
            ProcessingStage::StatsCalculation => &mut self.stats_log,
            _ => &mut self.processing_log,
        };
        
        if let Some(writer) = log_file {
            let _ = writer.write_all(log_line.as_bytes());
            let _ = writer.flush();
        }
    }
    
    pub fn set_stage(&mut self, stage: ProcessingStage) {
        self.current_stage = stage;
    }
    
    pub fn display_status_box(&self, status: StatusBox) {
        // Get the terminal width
        let terminal_width = match terminal_size() {
            Some((Width(w), Height(_))) => w as usize,
            None => 80,
        };
    
        // Calculate box width based on content
        let title_len = status.title.len();
        let max_content_width = status.stats.iter()
            .map(|(k, v)| k.len() + v.len() + 3) // +3 for separator and spacing
            .max()
            .unwrap_or(20);
    
        let content_width = std::cmp::max(max_content_width, title_len);
        let box_width = std::cmp::min(terminal_width - 4, content_width + 4);
    
        // Create timestamp for the status box
        let timestamp = Local::now().format("%H:%M:%S").to_string();
        let timestamp_display = format!(" [{}] ", timestamp);
        let timestamp_len = timestamp_display.len();
    
        // Create stage context for the status box
        let stage_context = match self.current_stage {
            ProcessingStage::Global => "[Global Context]",
            ProcessingStage::ConfigEntry => "[Config Entry]",
            ProcessingStage::VcfProcessing => "[VCF Processing]",
            ProcessingStage::VariantAnalysis => "[Variant Analysis]",
            ProcessingStage::CdsProcessing => "[CDS Processing]",
            ProcessingStage::StatsCalculation => "[Statistics]",
        };
        
        // Create the top border with timestamp
        let top_border = format!("┌{}{}{}┐", 
            "─".repeat((box_width - 2 - timestamp_len) / 2),
            timestamp_display,
            "─".repeat((box_width - 2 - timestamp_len + 1) / 2)
        );
    
        // Create the title bar with special formatting
        let padding = (box_width - 2 - title_len) / 2;
        let title_bar = format!("│{}{}{}│",
            " ".repeat(padding),
            status.title.bold(),
            " ".repeat(box_width - 2 - padding - title_len)
        );
        
        // Create the context bar
        let context_padding = (box_width - 2 - stage_context.len()) / 2;
        let context_bar = format!("│{}{}{}│",
            " ".repeat(context_padding),
            stage_context.dimmed(),
            " ".repeat(box_width - 2 - context_padding - stage_context.len())
        );
    
        // Create the divider
        let divider = format!("├{}┤", "─".repeat(box_width - 2));
    
        // Create the stats rows with improved formatting
        let mut stats_rows = Vec::new();
        for (key, value) in status.stats.iter() {
            let row = format!("│ {}: {}{} │",
                key.yellow(),
                value.white().bold(),
                " ".repeat(box_width - 5 - key.len() - value.len())
            );
            stats_rows.push(row);
        }
    
        // Create the bottom border
        let bottom_border = format!("└{}┘", "─".repeat(box_width - 2));
    
        // Log the status box creation
        self.log(LogLevel::Info, &format!("Displaying status box: {}", status.title));
    
        // Print the box with enhanced visuals
        println!("\n{}", top_border.cyan());
        println!("{}", title_bar.cyan());
        println!("{}", context_bar.cyan());
        println!("{}", divider.cyan());
        for row in stats_rows {
            println!("{}", row.cyan());
        }
        println!("{}\n", bottom_border.cyan());
    }
    
    pub fn finish_all(&mut self) {
        if let Some(bar) = &self.variant_bar {
            bar.finish_and_clear();
        }
        
        if let Some(bar) = &self.step_bar {
            bar.finish_and_clear();
        }
        
        if let Some(bar) = &self.entry_bar {
            bar.finish_and_clear();
        }
        
        if let Some(bar) = &self.global_bar {
            bar.finish_with_message(format!(
                "Processed {} entries in {:.2} seconds",
                self.total_entries,
                self.start_time.elapsed().as_secs_f64()
            ));
        }
        
        // Flush all log files
        if let Some(writer) = &mut self.processing_log {
            let _ = writer.flush();
        }
        if let Some(writer) = &mut self.variants_log {
            let _ = writer.flush();
        }
        if let Some(writer) = &mut self.transcripts_log {
            let _ = writer.flush();
        }
        if let Some(writer) = &mut self.stats_log {
            let _ = writer.flush();
        }
        
        // Print completion message
        println!("\n{}\n", "Analysis complete.".green().bold());
    }
    
    // Helper function to get the global progress tracker
    pub fn global() -> Arc<Mutex<ProgressTracker>> {
        PROGRESS_TRACKER.clone()
    }
}

// Helper functions for common operations
pub fn init_global_progress(total: usize) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.init_global_progress(total);
}

pub fn update_global_progress(current: usize, message: &str) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.update_global_progress(current, message);
}

pub fn init_entry_progress(entry_desc: &str, len: u64) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.init_entry_progress(entry_desc, len);
}

pub fn update_entry_progress(position: u64, message: &str) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.update_entry_progress(position, message);
}

pub fn finish_entry_progress(message: &str) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.finish_entry_progress(message);
}

pub fn init_step_progress(step_desc: &str, len: u64) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.init_step_progress(step_desc, len);
}

pub fn update_step_progress(position: u64, message: &str) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.update_step_progress(position, message);
}

pub fn finish_step_progress(message: &str) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.finish_step_progress(message);
}

pub fn init_variant_progress(desc: &str, len: u64) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.init_variant_progress(desc, len);
}

pub fn update_variant_progress(position: u64, message: &str) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.update_variant_progress(position, message);
}

pub fn finish_variant_progress(message: &str) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.finish_variant_progress(message);
}

pub fn create_spinner(message: &str) -> ProgressBar {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.spinner(message)
}

pub fn log(level: LogLevel, message: &str) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.log(level, message);
}

pub fn set_stage(stage: ProcessingStage) {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.set_stage(stage);
}

pub fn display_status_box(status: StatusBox) {
    let tracker = PROGRESS_TRACKER.lock();
    tracker.display_status_box(status);
}

pub fn finish_all() {
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.finish_all();
}
