use std::time::Duration;
use tokio::time::interval;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use log::{info, error, warn};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "dicai-provider")]
#[command(about = "DiCAI distributed inference provider node")]
struct Args {
    #[arg(long, default_value = "http://localhost:8080")]
    admin_url: String,
    
    #[arg(long, default_value = "provider-1")]
    id: String,
    
    #[arg(long, default_value = "16")]
    memory: u32,
    
    #[arg(long, default_value = "cpu")]
    backend: String,
    
    #[arg(long, default_value = "50051")]
    port: u16,
    
    #[arg(long, default_value = "localhost")]
    address: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProviderRegistration {
    id: String,
    address: String,
    port: u16,
    memory: u32,
    backend: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProviderHeartbeat {
    id: String,
    timestamp: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LayerAssignment {
    provider_id: String,
    model_id: String,
    layer_start: i32,
    layer_end: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegisterResponse {
    id: String,
    status: String,
}

struct ProviderNode {
    args: Args,
    client: Client,
    assignment: Option<LayerAssignment>,
}

impl ProviderNode {
    fn new(args: Args) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");
        
        ProviderNode {
            args,
            client,
            assignment: None,
        }
    }
    
    async fn register(&self) -> Result<RegisterResponse, reqwest::Error> {
        let registration = ProviderRegistration {
            id: self.args.id.clone(),
            address: self.args.address.clone(),
            port: self.args.port,
            memory: self.args.memory,
            backend: self.args.backend.clone(),
        };
        
        let url = format!("{}/api/v1/providers/register", self.args.admin_url);
        let response = self.client
            .post(&url)
            .json(&registration)
            .send()
            .await?;
        
        let result: RegisterResponse = response.json().await?;
        info!("Registered with admin: id={}, status={}", result.id, result.status);
        Ok(result)
    }
    
    async fn send_heartbeat(&self) -> Result<(), reqwest::Error> {
        let heartbeat = ProviderHeartbeat {
            id: self.args.id.clone(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        let url = format!("{}/api/v1/providers/heartbeat", self.args.admin_url);
        let response = self.client
            .post(&url)
            .json(&heartbeat)
            .send()
            .await?;
        
        if response.status().is_success() {
            info!("Heartbeat sent successfully");
        } else {
            warn!("Heartbeat failed: {}", response.status());
        }
        
        Ok(())
    }
    
    async fn run_heartbeat_loop(&self) {
        let mut ticker = interval(Duration::from_secs(5));
        
        loop {
            ticker.tick().await;
            if let Err(e) = self.send_heartbeat().await {
                error!("Heartbeat error: {}", e);
            }
        }
    }
    
    async fn start_grpc_server(&self) {
        // TODO: Implement gRPC server using tonic
        // For now, just log that we would start it
        info!("gRPC server would start on port {}", self.args.port);
        
        // Keep this task alive
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let args = Args::parse();
    
    info!("Starting DiCAI Provider Node...");
    info!("ID: {}", args.id);
    info!("Admin URL: {}", args.admin_url);
    info!("Backend: {}", args.backend);
    info!("Memory: {} GB", args.memory);
    
    let node = ProviderNode::new(args);
    
    // Register with admin
    match node.register().await {
        Ok(response) => {
            info!("Successfully registered: {}", response.id);
        }
        Err(e) => {
            error!("Failed to register: {}", e);
            std::process::exit(1);
        }
    }
    
    // Start heartbeat and gRPC server in parallel
    tokio::select! {
        _ = node.run_heartbeat_loop() => {
            error!("Heartbeat loop exited");
        }
        _ = node.start_grpc_server() => {
            error!("gRPC server exited");
        }
    }
}
