# About RNAPSEC
##  Interpretation and naming conventions of fields
- "index", "rnaphasep_index": index of RNAPhaSep. Experiments grouped together as the same data in RNAPhaSep have the same index in RNAPSEC.
- "rpsid": RNAPhaSep ID of the corresponding data. 
- "pmidlink": PubMed link of literature. 
- "components_type": Combination types of proteins and RNAs. components_type include "protein"(constructed by single protein), "RNA + protein"(constructed by a single RNA and RNA);
- "rna_sequence": RNA sequence is displayed in single letter sequence.
- "protein_sequence": Protein sequence is displayed in FASTA format. Post-translational modifications are not mentioned in sequence.
### Experiment related fields
- Represents whether the data is re-collected data." added" indicates that the data are new data that have been re-collected." edited" indicates that the data is a replication of  RNAPhaSep information, as experiments were conducted under one condition.
- "protein_conc": Value of protein concentration;"protein_unit": Unit of protein concentration.
- "rna_conc": Value of RNA concentration; "rna_unit": Unit of RNA concentration.
- "salt_conc": Value of salt; "salt_unit_name": Unit and Description of salt. Salt names and units are separated by "/", and if more than one salt is used, they are separated by ",".
- "pH": The pH of buffer solution.
- "morphology_add": Recollected annotation about phase morphology. "morphology_add" include "solute" (distributed state), "liquid" (formation of liquid-like condensates), "gel" (formation of gel-like condensates), and "solid" (formation of solid-like condensates).
- "description_article": Figure number of corresponding experiments in the article.
- "boundary": Indicates whether the phase state switched between the pre- and post-experimental conditions." y" indicates that the phase state has switched in the before/after experimental conditions." n" indicates that the phase state was the same as the pre- and post-experimental conditions.
- "temperature": Experimental temperature. All recorded in Celsius degrees. "RT" represents room temperature.
- The RNAPhaSep data are arranged from column X downwards; "temperature" to "morphology_add" columns are the newly organised data.

    **Note** Symbol "-" means none.
