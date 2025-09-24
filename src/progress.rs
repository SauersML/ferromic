use chrono::Local;
use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use terminal_size::{terminal_size, Height, Width};

// A global tracker that can be accessed from anywhere
pub static PROGRESS_TRACKER: Lazy<Arc<Mutex<ProgressTracker>>> =
    Lazy::new(|| Arc::new(Mutex::new(ProgressTracker::new())));

static PROGRESS_ALLOWED: Lazy<bool> = Lazy::new(|| {
    if let Ok(value) = std::env::var("FERROMIC_PROGRESS") {
        let normalized = value.to_ascii_lowercase();
        if matches!(normalized.as_str(), "0" | "false" | "off" | "no") {
            return false;
        }
        if matches!(normalized.as_str(), "1" | "true" | "on" | "yes") {
            return true;
        }
    }

    if std::env::var("PYTEST_CURRENT_TEST").is_ok() {
        return false;
    }

    std::io::stdout().is_terminal()
});

fn progress_enabled() -> bool {
    *PROGRESS_ALLOWED
}

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
    PcaAnalysis,
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
    total_entry_time: Duration,

    // Cache for reusable progress styles
    styles: HashMap<String, ProgressStyle>,
}

impl ProgressTracker {
    /// Creates a new ProgressTracker with multiple reusable styles
    pub fn new() -> Self {
        let multi_progress = MultiProgress::new();
        // Limit update frequency to reduce flickering in multi-threaded environments
        multi_progress.set_draw_target(ProgressDrawTarget::stdout_with_hz(12));

        // Create default log directory
        let log_dir = PathBuf::from("ferromic_logs");
        let _ = fs::create_dir_all(&log_dir);

        // Create reusable styles
        let mut styles = HashMap::new();

        // Global progress style with improved layout and ETA
        styles.insert(
            "global".to_string(),
            ProgressStyle::default_bar()
                .template("{spinner:.blue} {prefix:4} [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>4}/{len:<4} {percent:>3}% {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░"),
        );

        // Entry progress style with clear hierarchy
        styles.insert(
            "entry".to_string(),
            ProgressStyle::default_bar()
                .template("  {spinner:.green} {prefix:4} [{elapsed_precise}] {wide_bar:.green/white} {percent:>3}% {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░"),
        );

        // Step progress style with more detailed information
        styles.insert(
            "step".to_string(),
            ProgressStyle::default_bar()
                .template("    {spinner:.yellow} {prefix:4} [{elapsed_precise}] {wide_bar:.yellow/white} {pos:>6}/{len:<6} {percent:>3}% {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░"),
        );

        // Variant progress style with enhanced visualization
        styles.insert(
            "variant".to_string(),
            ProgressStyle::default_bar()
                .template("      {spinner:.magenta} {prefix:4} [{elapsed_precise}] {wide_bar:.magenta/white} {pos:>7}/{len:<7} {percent:>3}% {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░"),
        );

        // Spinner style with contextual information
        styles.insert(
            "spinner".to_string(),
            ProgressStyle::default_spinner()
                .template("{spinner:.bold.green} [{elapsed_precise}] {msg}")
                .expect("Spinner template error")
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );

        // VCF spinner for multi-threaded VCF loading
        styles.insert(
            "vcf_spinner".to_string(),
            ProgressStyle::default_spinner()
                .template("{spinner:.bold.green} [VCF] {elapsed_precise} {wide_msg}")
                .expect("Spinner template error")
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );

        // VCF byte progress bar for uncompressed files
        styles.insert(
            "vcf_bytes".to_string(),
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.bold.green} [VCF] {elapsed_precise} {wide_bar:.cyan/blue} {bytes:>10}/{total_bytes:<10} {wide_msg}"
                )
                .expect("Progress bar template error")
                .progress_chars("█▓▒░"),
        );

        // Add memory-intensive operation style for operations with heavy memory usage
        styles.insert(
            "memory_intensive".to_string(),
            ProgressStyle::default_bar()
                .template("    {spinner:.red} {prefix:4} [{elapsed_precise}] {wide_bar:.red/white} {pos:>7}/{len:<7} {percent:>3}% {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░"),
        );

        // Add IO-intensive operation style
        styles.insert(
            "io_intensive".to_string(),
            ProgressStyle::default_bar()
                .template("    {spinner:.blue} {prefix:4} [{elapsed_precise}] {wide_bar:.blue/white} {pos:>7}/{len:<7} {percent:>3}% {msg}")
                .expect("Progress bar template error")
                .progress_chars("█▓▒░"),
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
            total_entry_time: Duration::default(),
            styles,
        }
    }

    /// Helper function to create a log file if possible
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

    /// Attempt to find the topmost active bar for printing in-place text
    fn find_active_bar(&self) -> Option<&ProgressBar> {
        // Priority: variant > step > entry > global
        if let Some(ref bar) = self.variant_bar {
            return Some(bar);
        }
        if let Some(ref bar) = self.step_bar {
            return Some(bar);
        }
        if let Some(ref bar) = self.entry_bar {
            return Some(bar);
        }
        if let Some(ref bar) = self.global_bar {
            return Some(bar);
        }
        None
    }

    /// In-place printing that keeps output coordinated with progress bars
    fn inplace_print(&self, text: &str) {
        if let Some(bar) = self.find_active_bar() {
            bar.println(text);
        } else {
            // No active bar found, print directly
            println!("{}", text);
        }
    }

    /// Initialize global progress bar
    pub fn init_global_progress(&mut self, total: usize) {
        self.total_entries = total;
        self.current_entry = 0;
        self.total_entry_time = Duration::default();
        self.start_time = Instant::now();

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
        self.log(
            LogLevel::Info,
            &format!("Starting processing of {} config entries", total),
        );
    }

    /// Update global progress bar
    pub fn update_global_progress(&mut self, current: usize, message: &str) {
        if let Some(bar) = &self.global_bar {
            self.current_entry = current;
            bar.set_position(current as u64);
            bar.set_message(message.to_string());
        }
    }

    /// Initialize entry progress bar
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
        self.start_time = Instant::now();
        self.log(LogLevel::Info, &format!("Processing entry: {}", entry_desc));
    }

    /// Update entry progress
    pub fn update_entry_progress(&mut self, position: u64, message: &str) {
        if let Some(bar) = &self.entry_bar {
            bar.set_position(position);
            bar.set_message(message.to_string());
        }
    }

    /// Finish entry progress, show summary box
    pub fn finish_entry_progress(&mut self, message: &str, completed_units: usize) {
        // Calculate entry completion time and statistics
        let entry_name = self.entry_name.clone();
        let execution_time = self.start_time.elapsed();
        self.total_entry_time += execution_time;

        let total_entries = self.total_entries;
        let available = total_entries.saturating_sub(self.current_entry);
        let increment = completed_units.min(available);
        let new_total = self.current_entry + increment;

        // Format a detailed completion message with timing
        let completion_message = format!("{} (in {:.2}s)", message, execution_time.as_secs_f64());

        // Log detailed completion info
        self.log(
            LogLevel::Info,
            &format!(
                "Completed {} region(s) for {} ({}/{}) in {:.2}s",
                increment,
                entry_name,
                new_total,
                total_entries,
                execution_time.as_secs_f64()
            ),
        );

        // Finish entry progress bar with detailed message
        if let Some(bar) = &self.entry_bar {
            bar.finish_with_message(completion_message);
        }

        let progress_percentage = if total_entries > 0 {
            (new_total as f64 / total_entries as f64) * 100.0
        } else {
            100.0
        };
        let remaining_entries = total_entries.saturating_sub(new_total);

        let stats = vec![
            (String::from("Regions in this batch"), increment.to_string()),
            (
                String::from("Progress"),
                format!("{:.1}%", progress_percentage),
            ),
            (
                String::from("Time taken"),
                format!("{:.2}s", execution_time.as_secs_f64()),
            ),
            (
                String::from("Remaining regions"),
                remaining_entries.to_string(),
            ),
        ];

        let entry_box = StatusBox {
            title: format!(
                "Regions {}/{} complete ({})",
                new_total, total_entries, entry_name
            ),
            stats,
        };
        self.display_status_box(entry_box);

        // Also update global progress
        if let Some(bar) = &self.global_bar {
            self.current_entry = new_total;
            bar.set_position(self.current_entry as u64);

            let avg_time_per_entry = if self.current_entry > 0 {
                self.total_entry_time.as_secs_f64() / self.current_entry as f64
            } else {
                0.0
            };

            let est_remaining_time = avg_time_per_entry * remaining_entries as f64;

            bar.set_message(format!(
                "Completed {}/{} regions ({}) - {:.1}% complete - ~{:.0}s remaining",
                self.current_entry,
                total_entries,
                entry_name,
                progress_percentage,
                est_remaining_time
            ));
        }

        // Reset timer for next entry
        self.start_time = Instant::now();
    }

    /// Initialize step progress bar
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

    /// Update step progress
    pub fn update_step_progress(&mut self, position: u64, message: &str) {
        if let Some(bar) = &self.step_bar {
            bar.set_position(position);
            bar.set_message(message.to_string());
        }
    }

    /// Finish step progress
    pub fn finish_step_progress(&mut self, message: &str) {
        if let Some(bar) = &self.step_bar {
            bar.finish_with_message(message.to_string());
        }
    }

    /// Initialize variant progress bar
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
        self.log(
            LogLevel::Debug,
            &format!("Starting variant analysis: {}", desc),
        );
    }

    /// Update variant progress
    pub fn update_variant_progress(&mut self, position: u64, message: &str) {
        if let Some(bar) = &self.variant_bar {
            bar.set_position(position);
            bar.set_message(message.to_string());
        }
    }

    /// Finish variant progress
    pub fn finish_variant_progress(&mut self, message: &str) {
        if let Some(bar) = &self.variant_bar {
            bar.finish_with_message(message.to_string());
        }
    }

    /// Create and return a spinner (useful for short tasks)
    pub fn spinner(&mut self, message: &str) -> ProgressBar {
        // Spinner with stage context and better timing information
        let stage_indicator = match self.current_stage {
            ProcessingStage::Global => "[Global]",
            ProcessingStage::ConfigEntry => "[Entry]",
            ProcessingStage::VcfProcessing => "[VCF]",
            ProcessingStage::VariantAnalysis => "[Variant]",
            ProcessingStage::CdsProcessing => "[CDS]",
            ProcessingStage::StatsCalculation => "[Stats]",
            ProcessingStage::PcaAnalysis => "[PCA]",
        };

        let style = self.styles.get("spinner").cloned().unwrap_or_else(|| {
            ProgressStyle::default_spinner()
                .template(&format!(
                    "{{spinner:.bold.green}} {} {{msg}} [{{elapsed_precise}}]",
                    stage_indicator
                ))
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

    /// Create a spinner or progress bar dedicated to VCF loading operations
    pub fn create_vcf_progress(&mut self, total_bytes: Option<u64>, message: &str) -> ProgressBar {
        let progress = if let Some(total) = total_bytes {
            let bar = self.multi_progress.add(ProgressBar::new(total));
            let style = self
                .styles
                .get("vcf_bytes")
                .cloned()
                .unwrap_or_else(|| {
                    ProgressStyle::default_bar()
                        .template("{spinner:.bold.green} [VCF] {elapsed_precise} {wide_bar:.cyan/blue} {bytes}/{total_bytes} {msg}")
                        .expect("Progress bar template error")
                        .progress_chars("█▓▒░")
                });
            bar.set_style(style);
            bar
        } else {
            let spinner = self.multi_progress.add(ProgressBar::new_spinner());
            let style = self.styles.get("vcf_spinner").cloned().unwrap_or_else(|| {
                ProgressStyle::default_spinner()
                    .template("{spinner:.bold.green} [VCF] {elapsed_precise} {wide_msg}")
                    .expect("Spinner template error")
                    .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            });
            spinner.set_style(style);
            spinner.enable_steady_tick(Duration::from_millis(120));
            spinner
        };

        progress.set_message(message.to_string());
        self.log(
            LogLevel::Info,
            &format!("Started VCF operation: {}", message),
        );
        progress
    }

    /// Log a message to the stage-appropriate log file
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

    /// Change the current stage of processing
    pub fn set_stage(&mut self, stage: ProcessingStage) {
        self.current_stage = stage;
    }

    /// Display a fancy status box in-place
    pub fn display_status_box(&mut self, status: StatusBox) {
        // Get the terminal width
        let terminal_width = match terminal_size() {
            Some((Width(w), Height(_))) => w as usize,
            None => 80,
        };

        // Calculate box width based on content
        let title_len = status.title.len();
        let max_content_width = status
            .stats
            .iter()
            .map(|(k, v)| k.len() + v.len() + 3) // +3 for separator and spacing
            .max()
            .unwrap_or(20);

        let content_width = std::cmp::max(max_content_width, title_len);
        let box_width = std::cmp::min(terminal_width.saturating_sub(4), content_width + 4);

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
            ProcessingStage::PcaAnalysis => "[PCA Analysis]",
        };

        // Create the top border with timestamp
        let top_border = format!(
            "┌{}{}{}┐",
            "─".repeat((box_width.saturating_sub(2 + timestamp_len)) / 2),
            timestamp_display,
            "─".repeat((box_width.saturating_sub(2 + timestamp_len) + 1) / 2)
        );

        // Create the title bar with special formatting
        let padding = (box_width.saturating_sub(2 + title_len)) / 2;
        let title_bar = format!(
            "│{}{}{}│",
            " ".repeat(padding),
            status.title.bold(),
            " ".repeat(box_width.saturating_sub(2 + padding + title_len))
        );

        // Create the context bar
        let context_padding = (box_width.saturating_sub(2 + stage_context.len())) / 2;
        let context_bar = format!(
            "│{}{}{}│",
            " ".repeat(context_padding),
            stage_context.dimmed(),
            " ".repeat(box_width.saturating_sub(2 + context_padding + stage_context.len()))
        );

        // Create the divider
        let divider = format!("├{}┤", "─".repeat(box_width.saturating_sub(2)));

        // Create the stats rows with improved formatting
        let mut stats_rows = Vec::new();
        for (key, value) in status.stats.iter() {
            let row = format!(
                "│ {}: {}{} │",
                key.yellow(),
                value.white().bold(),
                " ".repeat(box_width.saturating_sub(5 + key.len() + value.len()))
            );
            stats_rows.push(row);
        }

        // Create the bottom border
        let bottom_border = format!("└{}┘", "─".repeat(box_width.saturating_sub(2)));

        // Log the status box creation
        self.log(
            LogLevel::Info,
            &format!("Displaying status box: {}", status.title),
        );

        // Build the multi-line string
        let mut final_box = String::new();
        final_box.push('\n');
        final_box.push_str(&top_border.cyan());
        final_box.push('\n');
        final_box.push_str(&title_bar.cyan());
        final_box.push('\n');
        final_box.push_str(&context_bar.cyan());
        final_box.push('\n');
        final_box.push_str(&divider.cyan());
        for row in stats_rows {
            final_box.push('\n');
            final_box.push_str(&row.cyan());
        }
        final_box.push('\n');
        final_box.push_str(&bottom_border.cyan());
        final_box.push('\n');

        // Print in-place
        self.inplace_print(&final_box);
    }

    /// Finish all progress bars and flush logs
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

        // Print completion message in-place
        let completion_str = format!("\n{}\n", "Analysis complete.".green().bold());
        self.inplace_print(&completion_str);
    }

    // Helper function to get the global progress tracker
    pub fn global() -> Arc<Mutex<ProgressTracker>> {
        PROGRESS_TRACKER.clone()
    }
}

// Helper functions for common operations
pub fn init_global_progress(total: usize) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.init_global_progress(total);
}

pub fn update_global_progress(current: usize, message: &str) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.update_global_progress(current, message);
}

pub fn init_entry_progress(entry_desc: &str, len: u64) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.init_entry_progress(entry_desc, len);
}

pub fn update_entry_progress(position: u64, message: &str) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.update_entry_progress(position, message);
}

pub fn finish_entry_progress(message: &str, completed_units: usize) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.finish_entry_progress(message, completed_units);
}

pub fn init_step_progress(step_desc: &str, len: u64) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.init_step_progress(step_desc, len);
}

pub fn update_step_progress(position: u64, message: &str) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.update_step_progress(position, message);
}

pub fn finish_step_progress(message: &str) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.finish_step_progress(message);
}

pub fn init_variant_progress(desc: &str, len: u64) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.init_variant_progress(desc, len);
}

pub fn update_variant_progress(position: u64, message: &str) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.update_variant_progress(position, message);
}

pub fn finish_variant_progress(message: &str) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.finish_variant_progress(message);
}

pub fn create_spinner(message: &str) -> ProgressBar {
    if !progress_enabled() {
        let bar = ProgressBar::hidden();
        bar.set_message(message.to_string());
        return bar;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.spinner(message)
}

pub fn create_vcf_progress(total_bytes: Option<u64>, message: &str) -> ProgressBar {
    if !progress_enabled() {
        let bar = ProgressBar::hidden();
        bar.set_message(message.to_string());
        return bar;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.create_vcf_progress(total_bytes, message)
}

pub fn log(level: LogLevel, message: &str) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.log(level, message);
}

pub fn set_stage(stage: ProcessingStage) {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.set_stage(stage);
}

/// Display status box without directly borrowing the mutable tracker.
pub fn display_status_box(status: StatusBox) {
    if !progress_enabled() {
        return;
    }
    // Get active progress bar and context information
    let active_bar_opt = {
        let mut tracker = PROGRESS_TRACKER.lock();

        // Get context
        let stage_context = match tracker.current_stage {
            ProcessingStage::Global => "[Global Context]",
            ProcessingStage::ConfigEntry => "[Config Entry]",
            ProcessingStage::VcfProcessing => "[VCF Processing]",
            ProcessingStage::VariantAnalysis => "[Variant Analysis]",
            ProcessingStage::CdsProcessing => "[CDS Processing]",
            ProcessingStage::StatsCalculation => "[Statistics]",
            ProcessingStage::PcaAnalysis => "[PCA Analysis]",
        };

        // Log the status box display
        tracker.log(
            LogLevel::Info,
            &format!("Displaying status box: {}", status.title),
        );

        // Create timestamp for the status box
        let timestamp = Local::now().format("%H:%M:%S").to_string();

        // Construct the status box content
        let mut content = String::new();
        content.push_str(&format!(
            "[{}] {} {}\n",
            timestamp, stage_context, status.title
        ));
        for (key, value) in status.stats.iter() {
            content.push_str(&format!("  {}: {}\n", key, value));
        }

        // Get the active bar if any (cloned to avoid borrowing issues)
        if let Some(bar) = tracker.find_active_bar() {
            Some((bar.clone(), content))
        } else {
            println!("\n{}", content);
            None
        }
    };

    // If we have an active bar, use its println method to avoid conflicts
    if let Some((bar, content)) = active_bar_opt {
        bar.println(format!("\n{}", content));
    }
}

pub fn finish_all() {
    if !progress_enabled() {
        return;
    }
    let mut tracker = PROGRESS_TRACKER.lock();
    tracker.finish_all();
}
