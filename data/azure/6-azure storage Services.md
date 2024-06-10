# Azure Storage

## Azure storage redundancy

### Locally Redundant Storage (LRS)

- Explanation: Stores data within a single physical location in the data center to protect against hardware failures.It replicates your data three times within a single data center in the primary region
- Durability in Nines: 11 nines (suitable for non-critical data)

### Zone-Redundant Storage (ZRS)

- Explanation:Zone-redundant storage (ZRS) replicates your Azure Storage data synchronously across three Azure availability zones in the primary region.
- Durability in Nines: 12 nines (good for critical regional data)

### Geo-Redundant Storage (GRS)

- Explanation: Replicates data to a secondary region that is hundreds of miles away from the primary region.GRS is similar to running LRS in two regions.
- Durability in Nines: 16 nines (recommended for essential data needing geographic redundancy)

### Geo-Zone-Redundant Storage (GZRS)

- Explanation: Combines the features of GRS and ZRS, replicating data across zones in the primary region and to a secondary region.
  ZRS is similar to running ZRS in the primary region and LRS in the secondary region.
- Durability in Nines: 16 nines (enhanced durability across zones and regions)

### Read-Access Geo-Redundant Storage (RA-GRS)

- Explanation: Same as GRS, but provides read-only access to the data in the secondary location.
- Durability in Nines: 16 nines (plus read access from secondary, ideal for applications needing fast read access during regional outages)

### Read-Access Geo-Zone-Redundant Storage (RA-GZRS)

- Explanation: Same as GZRS, but with read-only access to the data in the secondary location.
- Durability in Nines: 16 nines (plus read access from secondary, best for critical applications requiring maximum availability)

### Services

- **Azure Blobs**: Massively scalable object store for text and binary data, including support for big data analytics through Data Lake Storage Gen2.
- **Azure Files**: Managed file shares for cloud or on-premises deployments.
- **Azure Queues**: Messaging store for reliable messaging between application components.
- **Azure Disks**: Block-level storage volumes for Azure VMs.
- **Azure Tables**: NoSQL table option for structured, non-relational data.

### Benefits

- **Durable and Highly Available**: Data is safe even with hardware failures, with options to replicate data across data centers or geographical regions for added protection against local disasters.
- **Secure**: All data is encrypted, with detailed access controls.
- **Scalable**: Designed to meet the massive data storage and performance needs of modern applications.
- **Managed**: Azure handles maintenance, updates, and critical issues.
- **Accessible**: Accessible globally over HTTP/HTTPS, supported by multiple client libraries and tools.

### Specific Services Explained

#### Azure Blob Storage

- Ideal for storing massive amounts of unstructured data.
- Accessible via HTTP/HTTPS, supporting various client libraries.
- **Blob Storage Tiers**: Different tiers (Hot, Cool, Cold, Archive) to manage costs based on frequency of access and retention period.

#### Azure Files

- Offers fully managed file shares that are accessible via SMB or NFS protocols.
- Can be mounted concurrently by cloud or on-premises deployments.
- **Key Benefits**: Shared access, fully managed, support for scripting and tooling, built-in resiliency.

#### Azure Queues

- Service for storing large numbers of messages, accessible globally.
- Useful for creating a backlog of work to process asynchronously.

#### Azure Disks

- Provides block-level storage volumes for Azure VMs, managed by Azure for greater resiliency.

#### Azure Tables

- NoSQL datastore for storing large amounts of structured, non-relational data.
- Ideal for applications requiring a hybrid or multi-cloud setup.

### Blob storage tiers

- **Hot access tier**: Optimized for storing data that is accessed frequently (for example, images for your website).
- **Cool access tier**: Optimized for data that is infrequently accessed and stored for at least 30 days (for example, invoices for your customers).
- **Cold access tier**: Optimized for storing data that is infrequently accessed and stored for at least 90 days.
- **Archive access tier**: Appropriate for data that is rarely accessed and stored for at least 180 days, with flexible latency requirements (for example, long-term backups).

The following considerations apply to the different access tiers:
Hot and cool access tiers can be set at the account level. The cold and archive access tiers aren't available at the account level.
Hot, cool, cold, and archive tiers can be set at the blob level, during or after upload.
Data in the cool and cold access tiers can tolerate slightly lower availability, but still requires high durability, retrieval latency, and throughput characteristics similar to hot data. For cool and cold data, a lower availability service-level agreement (SLA) and higher access costs compared to hot data are acceptable trade-offs for lower storage costs.
Archive storage stores data offline and offers the lowest storage costs, but also the highest costs to rehydrate and access data.

### Accessing Storage

- Objects in blob storage can be accessed globally via URLs, the Azure Storage REST API, Azure PowerShell, Azure CLI, or client libraries.
- Azure Files can be accessed via SMB or NFS, with Azure File Sync for caching on Windows Servers.
- Azure Tables and Queues support authenticated calls from inside and outside Azure.

## Azure Data Migration Services

### Azure Migrate

Azure Migrate is a comprehensive service designed to facilitate the transition from on-premises environments to the cloud. Key features include:

- **Unified Migration Platform**: Central portal for initiating, managing, and tracking migrations.
- **Integrated Tools**:
  - **Azure Migrate: Discovery and Assessment**: Discover and assess on-premises servers (VMware, Hyper-V, physical servers) for migration.
  - **Azure Migrate: Server Migration**: Migrate virtualized servers and public cloud VMs to Azure.
  - **Data Migration Assistant**: Assess SQL Servers for migration, highlighting potential issues and beneficial features.
  - **Azure Database Migration Service**: Migrate on-premises databases to Azure SQL solutions.
  - **Azure App Service Migration Assistant**: Assess and migrate .NET and PHP web apps to Azure App Service.

### Azure Data Box

Azure Data Box provides a physical solution for transferring large volumes of data to Azure, ideal for scenarios with limited network connectivity or large data sets exceeding 40 TBs. The process includes:

- **Ordering and Setup**: Order through the Azure portal, set up via local web UI, and connect to your network.
- **Secure Data Transfer**: Data is stored on an 80 TB capacity device, shipped with robust security and tracking.
- **Use Cases**:
  - **One-time Migration**: Large-scale data transfers to Azure.
  - **Initial Bulk Transfer and Periodic Uploads**: Combination of initial large transfer followed by regular smaller updates.
  - **Exporting Data from Azure**: For disaster recovery, security requirements, or transitioning to other services.
- **Security and Compliance**: Post-transfer, disks are wiped clean following NIST 800-88r1 standards.

These services ensure secure, efficient, and scalable solutions for migrating vast amounts of data to Azure, supporting both real-time and asynchronous migration strategies.

## Azure File Movement Tools

### AzCopy

- **Description**: A command-line utility for copying blobs or files to or from Azure storage accounts.
- **Features**:
  - Upload, download, and copy files between storage accounts.
  - One-direction synchronization from a designated source to destination.
  - Configurable for use with other cloud providers.
- **Important**: Synchronization is one-directional; it does not support bi-directional sync based on timestamps or other metadata.

### Azure Storage Explorer

- **Description**: A graphical interface for managing files and blobs in Azure Storage Accounts.
- **Platform Support**: Available on Windows, macOS, and Linux.
- **Capabilities**:
  - Facilitates uploading to and downloading from Azure.
  - Enables moving files between storage accounts.
  - Utilizes AzCopy on the backend for file and blob management tasks.

### Azure File Sync

- **Description**: Synchronizes local Windows file servers with Azure Files, enhancing flexibility, performance, and compatibility.
- **Features**:
  - Bi-directional synchronization between local servers and Azure.
  - Supports various protocols on Windows Server like SMB, NFS, and FTPS for local data access.
  - Enables the creation of multiple caches worldwide.
  - Allows quick replacement of failed local servers.
  - Configures cloud tiering for optimal file availability and storage efficiency.

These tools offer tailored solutions for managing file transfers and interactions within Azure's ecosystem, from individual files to larger groups, across various platforms and locations.
