# Microsoft Azure Overview

## What is Microsoft Azure?

Azure is a comprehensive set of cloud services that allows you to build, manage, and deploy applications on a global network. It provides the tools and frameworks you need to meet both current and future business challenges, promoting innovation with robust support for AI, machine learning, and more.

## Key Offerings

- **Limitless Innovation**: Build advanced, intelligent applications to elevate your business, utilizing industry-leading AI and cloud services.
- **Unified Management**: Seamlessly manage infrastructure, data, analytics, and AI solutions across an integrated platform.
- **Trusted Platform**: Innovate with confidence on a secure and responsible platform dedicated to enterprise needs.

## Capabilities of Azure

- **Versatile Services**: Over 100 services to run virtual machines, develop smart bots, manage data, and more.
- **AI and Machine Learning**: Create applications that interact naturally with users through enhanced cognitive services.
- **Dynamic Storage Solutions**: Address massive data needs with scalable and flexible storage options.

## Getting Started with Azure

- **Azure Accounts and Subscriptions**: Essential for creating and managing Azure resources. Supports multiple subscriptions under a single account for organizational flexibility.
- **Free Accounts**:
  - **Standard Free Account**: Includes 12 months of popular free services, a 30-day credit, and access to over 25 always-free services.
  - **Student Account**: Offers $100 credit and free access to certain services and developer tools, no credit card required.

## Azure Physical and Management Infrastructure

- **Datacenters and Regions**: Global network of datacenters organized into regions and availability zones for resilience and redundancy.
- **Availability Zones**: Independent datacenters within a region, equipped with their own power, cooling, and networking, designed to support continuity and disaster recovery.
- **Region Pairs**: Enhances disaster recovery by pairing regions geographically to ensure at least one is operational during regional outages.

## Azure Hierarchical Structure

- **Resources**: The basic units like VMs, databases, which are grouped into:
- **Resource Groups**: Containers that hold related resources for easier management and deployment.
- **Subscriptions**: Billing and management units that contain one or more resource groups.
- **Management Groups**: Collections of subscriptions that provide policy and governance at scale.

## Usage Scenarios

Azure's flexible structure supports various scenarios, including isolated environments for development, testing, security, and compliance, through separate subscriptions or management groups for different organizational needs.

This summary encapsulates the essence of Azure, emphasizing its infrastructure, management hierarchy, and diverse capabilities, tailored for quick understanding or as an introduction for new users or stakeholders.

## Explanation

- **Azure Account**: The top-level entity, under which all other structures are organized.
- **Management Groups**: Used to manage access, policy, and compliance across multiple subscriptions. Management groups can contain multiple subscriptions.
- **Subscriptions**: Containers for billing and policy application. Each subscription can host multiple resource groups.
- **Resource Groups**: Collections of resources that share the same lifecycle, permissions, and policies.
- **Resources**: Individual instances of services such as virtual machines, databases, etc., contained within resource groups.

This diagram represents how various components within # Azure Hierarchical Structure

- **Azure Account**
  - **Management Group 1**
    - **Subscription A**
      - **Resource Group I**
        - Resource 1
        - Resource 2
      - **Resource Group II**
        - Resource 3
        - Resource 4
    - **Subscription B**
      - **Resource Group III**
        - Resource 5
        - Resource 6
      - **Resource Group IV**
        - Resource 7
        - Resource 8
  - **Management Group 2**
    - **Subscription C**
      - **Resource Group V**
        - Resource 9
        - Resource 10
      - **Resource Group VI**
        - Resource 11
        - Resource 12

## Azure Support Plans
| Support Plan        | Cost               | Scope of Support                                          | Available Services                                     |
|---------------------|--------------------|-----------------------------------------------------------|--------------------------------------------------------|
| **Basic**           | Free               | Billing support and subscription management only.         | Self-help resources and community support.             |
| **Developer**       | Approximately $29/month | Best for trial and non-production environments.         | Unlimited 24/7 access to technical support via email for non-critical issues, access to full set of Azure Advisor recommendations. |
| **Standard**        | Starting at $100/month | Recommended for production workloads.                   | Everything in Developer, plus 24/7 access to technical support with faster response times for critical issues, and proactive guidance from Azure Advisor. |
| **Professional Direct** | Starting at $1000/month | Ideal for business-critical dependence on Azure.       | Everything in Standard, plus faster response times, access to Azure Rapid Response, Azure Event Management, and ProDirect Delivery Manager. |
| **Premier**         | Custom pricing     | Enterprises that require a higher level of partnership.  | All previous benefits plus a designated Technical Account Manager, Azure Solutions best practices, personalized proactive support, and more. |

### Key Features:
- **Response Times:** Varies by plan, with the fastest responses available under the Premier plan.
- **Technical Support:** Ranges from email support for non-critical issues (Developer) to 24/7 phone support for all issues including critical (Premier).
- **Proactive Support:** Available from the Standard plan upwards, including full Azure Advisor recommendations.
- **Direct Contact:** Available in the Professional Direct and Premier plans, offering direct access to Azure experts.

### Additional Notes:
- Pricing can vary based on region and specific needs.
- More comprehensive services like a designated Technical Account Manager and on-demand engineers are provided in the higher-tier plans to ensure enterprise-grade support.

This table offers a structured comparison to help organizations choose the appropriate Azure support plan based on their operational needs and criticality of their Azure deployments.
