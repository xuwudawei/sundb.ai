import logging
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)

filename="/Users/apple/Desktop/Tsinghua/Research/gpu_sundb.ai/sundb.ai_single_knowledge_base_version/data/uploads/01953722bd687241a27146f923c2e3dd/1741152848-019564cc0bad751eae0e2e518b747149.pdf"

print("About to call partition_pdf on file: %s", filename)
raw_pdf_elements = partition_pdf(
    filename=filename,
    extract_images_in_pdf=True,
    pdf_infer_table_structure=True,
    extract_image_block_types=["Image", "Table"],  
    extract_image_block_to_payload=False,   
    strategy="auto",
    extract_image_block_output_dir="/Users/apple/Desktop/Tsinghua/Research/gpu_sundb.ai/sundb.ai_single_knowledge_base_version/data",
)
print("partition_pdf returned %d elements", len(raw_pdf_elements))
