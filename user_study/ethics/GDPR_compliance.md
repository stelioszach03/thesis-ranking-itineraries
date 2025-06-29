# GDPR Compliance Checklist

## Lawful Basis for Processing
- [x] **Consent**: Explicit consent obtained via consent form
- [x] **Legitimate Interest**: Academic research with societal benefits
- [x] **Public Task**: University research mission

## Data Minimization
- [x] Only collect data necessary for research objectives
- [x] Anonymous participant IDs (P001-P030)
- [x] No unnecessary personal identifiers
- [x] IP addresses not logged

## Data Collected
1. **Demographics** (anonymized)
   - Age range (not exact age)
   - Travel frequency category
   - NYC familiarity level
   
2. **Interaction Data**
   - Task completion times
   - Algorithm selections
   - Itinerary modifications
   - Click patterns
   
3. **Feedback Data**
   - Questionnaire responses
   - Interview transcripts (anonymized)
   - Screen recordings (with consent, faces blurred)

## Data Subject Rights

### 1. Right to Access (Article 15)
- Participants can request all their data
- Response within 30 days
- Format: JSON export

### 2. Right to Rectification (Article 16)
- Corrections to demographic data allowed
- Task data immutable for research integrity

### 3. Right to Erasure (Article 17)
- Full deletion available until analysis begins
- After analysis: anonymized data retained

### 4. Right to Restrict Processing (Article 18)
- Data can be quarantined pending resolution

### 5. Right to Data Portability (Article 20)
- Machine-readable format provided
- Standard JSON schema

### 6. Right to Object (Article 21)
- Withdrawal form available
- Clear opt-out process

## Technical Measures

### Security
- [x] Encrypted storage (AES-256)
- [x] Encrypted transmission (TLS 1.3)
- [x] Access logging
- [x] Regular backups

### Privacy by Design
- [x] Pseudonymization from collection
- [x] Automatic data expiry (5 years)
- [x] Minimal data retention
- [x] No third-party sharing

## Data Processing

### Storage
- Location: EU-based servers / University secure storage
- Encryption: At rest and in transit
- Access: Role-based (PI, approved researchers only)
- Backup: Weekly, encrypted, separate location

### Retention
- Active study: Until analysis complete
- Archive: 5 years per university policy
- Deletion: Secure wiping (NIST 800-88)

### International Transfers
- No transfers outside EU/EEA
- If required: Standard Contractual Clauses

## Breach Response

### Detection
- Automated monitoring
- Regular audits
- Incident logging

### Response (within 72 hours)
1. Contain breach
2. Assess scope
3. Notify DPO
4. Notify supervisory authority if required
5. Notify affected participants if high risk

## Data Protection Officer

Name: [DPO Name]
Email: dpo@university.edu
Phone: +30 XXX XXX XXXX

## Compliance Documentation

- [x] Privacy Impact Assessment completed
- [x] Records of Processing Activities (Article 30)
- [x] Consent records maintained
- [x] Training logs for research team

## Audit Trail

All processing activities logged:
```
{
  "timestamp": "ISO-8601",
  "action": "data_access|data_modify|data_export|data_delete",
  "participant_id": "P001",
  "researcher_id": "R001",
  "justification": "string",
  "ip_address": "hashed"
}
```

## Third-Party Processors

| Processor | Purpose | Location | Safeguards |
|-----------|---------|----------|------------|
| University IT | Storage | Greece/EU | Internal agreement |
| Cloud Backup | Backup | Ireland/EU | DPA signed |

## Regular Reviews

- [ ] Q1 2025: Initial compliance check
- [ ] Q2 2025: Post-study review
- [ ] Q3 2025: Annual audit
- [ ] Q4 2025: Policy updates

## Contact for Data Protection Queries

Research Ethics Committee
ethics@university.edu
+30 XXX XXX XXXX

Data Protection Officer
dpo@university.edu
+30 XXX XXX XXXX