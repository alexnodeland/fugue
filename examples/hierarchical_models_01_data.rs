use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Debug, Clone)]
struct SchoolData {
    school_id: usize,
    name: String,
    n_students: usize,
    scores: Vec<f64>,
    sample_mean: f64,
}

fn generate_school_data() -> Vec<SchoolData> {
    let mut rng = StdRng::seed_from_u64(2024);
    
    // True population parameters (unknown to our model)
    let true_pop_mean = 78.0;     // Population average
    let true_between_std = 6.0;   // Between-school variation
    let true_within_std = 12.0;   // Within-school variation (student-level)
    
    let school_names = vec![
        "Lincoln Elementary", "Washington Middle", "Roosevelt High", 
        "Jefferson Academy", "Madison Prep", "Monroe Charter",
        "Adams Elementary", "Hamilton High", "Franklin Middle",
        "Wilson Academy", "Garfield Elementary", "Kennedy High"
    ];
    
    let mut schools = Vec::new();
    
    for (id, name) in school_names.into_iter().enumerate() {
        // Sample true school effect
        let true_school_effect = Normal::new(true_pop_mean, true_between_std)
            .unwrap()
            .sample(&mut rng);
        
        // Varying sample sizes (realistic for different schools)
        let n_students = match id {
            0..=2 => 15 + (id * 3),      // Small schools: 15-21 students
            3..=6 => 30 + (id * 5),      // Medium schools: 45-60 students  
            _ => 80 + (id * 10),         // Large schools: 80+ students
        };
        
        // Generate student scores for this school
        let scores: Vec<f64> = (0..n_students)
            .map(|_| {
                Normal::new(true_school_effect, true_within_std)
                    .unwrap()
                    .sample(&mut rng)
                    .max(0.0)  // Ensure non-negative scores
                    .min(100.0) // Cap at 100
            })
            .collect();
        
        let sample_mean = scores.iter().sum::<f64>() / scores.len() as f64;
        
        schools.push(SchoolData {
            school_id: id,
            name: name.to_string(),
            n_students,
            scores,
            sample_mean,
        });
    }
    
    schools
}

fn explore_school_data() {
    let schools = generate_school_data();
    
    println!("🏫 School District Test Score Analysis");
    println!("=====================================");
    
    let total_students: usize = schools.iter().map(|s| s.n_students).sum();
    let overall_mean = schools.iter()
        .flat_map(|s| &s.scores)
        .sum::<f64>() / total_students as f64;
    
    println!("📊 District Overview:");
    println!("  Number of schools: {}", schools.len());
    println!("  Total students: {}", total_students);
    println!("  Overall mean score: {:.2}", overall_mean);
    
    println!("\n🏫 School-by-School Breakdown:");
    println!("  School                | N   | Mean  | Min   | Max   ");
    println!("  ---------------------|-----|-------|-------|-------");
    
    for school in &schools {
        let min_score = school.scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_score = school.scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("  {:20} | {:3} | {:5.1} | {:5.1} | {:5.1}", 
                 school.name, school.n_students, school.sample_mean, min_score, max_score);
    }
    
    // Show sampling variability issue
    let small_schools: Vec<&SchoolData> = schools.iter().filter(|s| s.n_students < 30).collect();
    let large_schools: Vec<&SchoolData> = schools.iter().filter(|s| s.n_students > 60).collect();
    
    if !small_schools.is_empty() && !large_schools.is_empty() {
        let small_var = small_schools.iter()
            .map(|s| s.sample_mean)
            .map(|x| (x - overall_mean).powi(2))
            .sum::<f64>() / small_schools.len() as f64;
        
        let large_var = large_schools.iter()
            .map(|s| s.sample_mean)
            .map(|x| (x - overall_mean).powi(2))
            .sum::<f64>() / large_schools.len() as f64;
        
        println!("\n📏 Sampling Variability:");
        println!("  Small schools (<30 students) variance: {:.2}", small_var);
        println!("  Large schools (>60 students) variance: {:.2}", large_var);
        
        if small_var > large_var * 1.5 {
            println!("  🎯 Small schools show higher variability - perfect case for hierarchical modeling!");
        }
    }
}

fn main() {
    explore_school_data();
}