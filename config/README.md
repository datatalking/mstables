# Configuration Files

This directory contains configuration templates for machine-specific settings.

## Security Notice

⚠️ **IMPORTANT**: These configuration files contain sensitive information (IP addresses, hostnames, usernames) and are excluded from git via `.gitignore`.

**DO NOT** commit actual configuration files to the repository.

## Setup

1. Copy the template files to create your actual configuration:
   ```bash
   cp config/fleet_config.json.template config/fleet_config.json
   cp config/machine_config.json.template config/machine_config.json
   ```

2. Edit the configuration files with your actual machine information:
   - Replace placeholder hostnames with your actual machine hostnames
   - Replace placeholder IP addresses with your actual machine IPs
   - Replace placeholder usernames with your actual usernames
   - Update MAC addresses if needed
   - Update descriptions as needed

3. Verify the files are in `.gitignore`:
   ```bash
   git check-ignore config/fleet_config.json config/machine_config.json
   ```

## Files

### `fleet_config.json.template`
Template for SLURM distributed computing fleet configuration.
- Defines machines available for job distribution
- Includes hostname, IP, CPU, memory, GPU availability
- Includes SSH user and key path

### `machine_config.json.template`
Template for network device discovery configuration.
- Defines network devices and their properties
- Includes hostname, IP, MAC address
- Includes usernames for each machine
- Includes gateway/router information

## Example

After copying and editing, your `fleet_config.json` might look like:

```json
{
  "machines": {
    "workstation_1": {
      "hostname": "workstation-1.local",
      "ip": "192.168.1.100",
      "cpus": 8,
      "memory_gb": 64,
      "gpu_available": true,
      "ssh_user": "your_username",
      "ssh_key_path": "/path/to/your/ssh/key",
      "description": "Primary workstation"
    }
  }
}
```

## Troubleshooting

If you see warnings about missing config files:
1. Make sure you've copied the templates to the actual config files
2. Verify the file paths are correct
3. Check file permissions (should be readable)
4. Verify JSON syntax is valid

## Migration from Hardcoded Values

If you're migrating from code with hardcoded machine information:
1. Copy the template files
2. Extract your machine information from the old code
3. Update the template files with your actual information
4. The code will automatically load from these config files

