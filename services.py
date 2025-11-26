import os
import io
import base64
import json
from PIL import Image
from openai import OpenAI
from prompts import PROMPTS

from stl import mesh
import cv2
import numpy as np

# Initialize OpenAI Client
# Ensure OPENAI_API_KEY is set in your environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ImageService:
    def __init__(self, upload_dir="static/uploads", processed_dir="static/processed", models_dir="static/models"):
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def convert_to_png(self, file_content: bytes, filename: str) -> str:
        """Converts uploaded image bytes to PNG and saves it."""
        try:
            image = Image.open(io.BytesIO(file_content))
            name_no_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(self.upload_dir, f"{name_no_ext}.png")
            
            image.convert('RGBA').save(save_path)
            print(f"Converted: {filename} -> {save_path}")
            return save_path
        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")

    def encode_image(self, image_path: str) -> str:
        """Encodes an image file to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def analyze_proportions(self, image_path: str, model="gpt-4o") -> dict:
        """Analyzes keychain image to extract dimensional proportions."""
        image_base64 = self.encode_image(image_path)
        
        function_schema = [
            {
                'name': 'extract_keychain_proportions',
                'description': 'Extract the physical dimensional proportions and shape information from a keychain image',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'width': {
                            'type': 'number',
                            'description': 'Width dimension (longest vertical extent). Use 1.0 as baseline for normalization.'
                        },
                        'length': {
                            'type': 'number',
                            'description': 'Length dimension (longest horizontal extent). Relative to width.'
                        },
                        'thickness': {
                            'type': 'number',
                            'description': 'Thickness/depth from front to back surface.'
                        },
                        'complexity': {
                            'type': 'string',
                            'enum': ['simple', 'moderate', 'complex'],
                            'description': 'Visual complexity of the shape.'
                        },
                    },
                    'required': ['width', 'length', 'thickness', 'complexity']
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPTS.RATIO_ANALYSIS},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                }
                            }
                        ]
                    }
                ],
                functions=function_schema,
                function_call={"name": "extract_keychain_proportions"}
            )

            message = response.choices[0].message
            
            if message.function_call:
                args = json.loads(message.function_call.arguments)
                return {
                    "success": True,
                    "data": args,
                    "ratio_string": f"{args['length']}:{args['width']}:{args['thickness']}"
                }
            
            return {"success": False, "error": "No function call returned"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_silhouette(self, image_path: str, model="gpt-image-1") -> str:
        """Generates a silhouette from the original image."""
        try:
            result = client.images.edit(
                model=model,
                image=open(image_path, "rb"),
                prompt=PROMPTS.SILHOUETTE_EXTRACTION,
                n=1,
                # size="1024x1024"
            )
            
            # Handle Base64 response
            image_data = result.data[0]
            
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                image_bytes = base64.b64decode(image_data.b64_json)
            elif hasattr(image_data, 'url'):
                # If API returns URL, download it (simplified here, assumes b64 request)
                # To force b64, usually need response_format="b64_json" in call
                # For this demo, we will assume the prompt implies we want the image data
                # If using standard DALL-E 2, add response_format="b64_json" to the call above
                raise ValueError("Please configure OpenAI call to return b64_json")

            filename = os.path.basename(image_path).split('.')[0] + '_silhouette.png'
            output_path = os.path.join(self.processed_dir, filename)
            
            return self._download_or_save_image(image_data, output_path)

        except Exception as e:
            print(f"Error generating silhouette: {e}")
            raise e

    def edit_silhouette(self, image_path: str="", model="gpt-image-1", instructions: str="") -> str:
        """Edits a silhouette based on red marks and instructions."""

        if not image_path:
            combined_prompt = PROMPTS.SILHOUETTE_TEXT_PROMPT.format(user_instruction=instructions)
        else:
            combined_prompt = PROMPTS.SILHOUETTE_EDIT_PROMPT.format(user_instruction=instructions)
        
        try:
            if not image_path:
                result = client.images.edit(
                    model="gpt-image-1",
                    image=open(image_path, "rb"),
                    prompt=combined_prompt,
                    n=1,
                )
            else:
                result = client.images.edit(
                    model=model,
                    # image=open(image_path, "rb"),
                    prompt=combined_prompt,
                    n=1,
                )
            
            filename = os.path.basename(image_path).split('.')[0] + '_updated.png'
            output_path = os.path.join(self.processed_dir, filename)
            
            image_bytes = base64.b64decode(result.data[0].b64_json)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
                
            return output_path
        except Exception as e:
            raise ValueError(f"Error editing silhouette: {str(e)}")

    def _download_or_save_image(self, image_data, output_path):
        """Helper to handle OpenAI image response (URL or B64)."""
        image_b64 = image_data.b64_json
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return output_path
    
    def convert_depth_to_stl(
        self, 
        image_path: str, 
        depth_div_width: float,
        aspect_ratio: float = 0.2
    ) -> str:
        """
        Convert a depth map image to an STL 3D model.
        
        Args:
            image_path: Path to the depth map image
            depth_div_width: Ratio of desired depth to width (e.g., 0.5)
            aspect_ratio: Height to width ratio for the model (default: 1.0)
            
        Returns:
            Path to the generated STL file
        """
        # Read the image
        im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if im is None:
            raise ValueError(f"Failed to read image from {image_path}")
        
        # Process image array
        im_array = np.array(im)
        im_array = 255 - im_array
        im_array = np.rot90(im_array, -1, (0, 1))
        
        mesh_size = [im_array.shape[0], im_array.shape[1]]
        mesh_max = np.max(im_array)
        
        if mesh_max == 0:
            raise ValueError("Image contains no depth information (all pixels are black)")
        
        # Scale mesh based on depth information
        if len(im_array.shape) == 3:
            # Color image - use first channel
            scaled_mesh = mesh_size[0] * depth_div_width * im_array[:, :, 0] / mesh_max
        else:
            # Grayscale image
            scaled_mesh = mesh_size[0] * depth_div_width * im_array / mesh_max
        
        # Create mesh
        mesh_shape = mesh.Mesh(
            np.zeros((mesh_size[0] - 1) * (mesh_size[1] - 1) * 2, dtype=mesh.Mesh.dtype)
        )
        
        # Generate triangles for the mesh
        for i in range(0, mesh_size[0] - 1):
            for j in range(0, mesh_size[1] - 1):
                mesh_num = i * (mesh_size[1] - 1) + j
                
                # Apply aspect ratio to i coordinate (height)
                i_scaled = i * aspect_ratio
                i1_scaled = (i + 1) * aspect_ratio
                
                # First triangle
                mesh_shape.vectors[2 * mesh_num][2] = [i_scaled, j, scaled_mesh[i, j]]
                mesh_shape.vectors[2 * mesh_num][1] = [i_scaled, j + 1, scaled_mesh[i, j + 1]]
                mesh_shape.vectors[2 * mesh_num][0] = [i1_scaled, j, scaled_mesh[i + 1, j]]
                
                # Second triangle
                mesh_shape.vectors[2 * mesh_num + 1][0] = [i1_scaled, j + 1, scaled_mesh[i + 1, j + 1]]
                mesh_shape.vectors[2 * mesh_num + 1][1] = [i_scaled, j + 1, scaled_mesh[i, j + 1]]
                mesh_shape.vectors[2 * mesh_num + 1][2] = [i1_scaled, j, scaled_mesh[i + 1, j]]
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_name}_3d.stl"
        output_path = os.path.join("static/models", output_filename)
        
        # Save mesh to file
        mesh_shape.save(output_path)
        return output_path
    
    
    # def convert_depth_to_stl(
    #     self,
    #     image_path: str,
    #     depth_div_width: float,
    #     aspect_ratio: float = 1.0,
    #     reduce_factor: float = 0.5,
    #     smooth: bool = True
    # ) -> str:
    #     """
    #     Convert a depth map image to a solid STL 3D model.

    #     Args:
    #         image_path: Path to the depth map image
    #         depth_div_width: Ratio of desired depth to width
    #         aspect_ratio: height scaling
    #         reduce_factor: resolution scaling factor (<1 reduces file size)
    #         smooth: apply gaussian blur to reduce noise

    #     Returns:
    #         Path to generated STL file
    #     """
    #     import os
    #     import cv2
    #     import numpy as np
    #     from stl import mesh

    #     # ------------ 1. Load image ------------
    #     im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #     if im is None:
    #         raise ValueError(f"Failed to read image from {image_path}")

    #     # Reduce image resolution if needed
    #     if reduce_factor < 1.0:
    #         new_w = int(im.shape[1] * reduce_factor)
    #         new_h = int(im.shape[0] * reduce_factor)
    #         im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    #     # Smooth to reduce noise
    #     if smooth:
    #         im = cv2.GaussianBlur(im, (7, 7), 0)

    #     # Convert to array
    #     im_array = 255 - np.array(im)
    #     im_array = np.rot90(im_array, -1, (0, 1))

    #     mesh_size = [im_array.shape[0], im_array.shape[1]]
    #     mesh_max = np.max(im_array)
    #     if mesh_max == 0:
    #         raise ValueError("Image contains no depth information")

    #     # ------------ 2. Build height map ------------
    #     if len(im_array.shape) == 3:
    #         height_map = mesh_size[0] * depth_div_width * im_array[:, :, 0] / mesh_max
    #     else:
    #         height_map = mesh_size[0] * depth_div_width * im_array / mesh_max

    #     H, W = mesh_size

    #     # Number of triangles for the top surface
    #     top_tri_count = (H - 1) * (W - 1) * 2

    #     # ------------ 3. Build TOP surface ------------
    #     top_mesh = mesh.Mesh(np.zeros(top_tri_count, dtype=mesh.Mesh.dtype))

    #     idx = 0
    #     for i in range(H - 1):
    #         for j in range(W - 1):

    #             i0 = i * aspect_ratio
    #             i1 = (i + 1) * aspect_ratio

    #             # Triangle 1
    #             top_mesh.vectors[idx][0] = [i0, j, height_map[i, j]]
    #             top_mesh.vectors[idx][1] = [i0, j+1, height_map[i, j+1]]
    #             top_mesh.vectors[idx][2] = [i1, j, height_map[i+1, j]]
    #             idx += 1

    #             # Triangle 2
    #             top_mesh.vectors[idx][0] = [i1, j+1, height_map[i+1, j+1]]
    #             top_mesh.vectors[idx][1] = [i0, j+1, height_map[i, j+1]]
    #             top_mesh.vectors[idx][2] = [i1, j, height_map[i+1, j]]
    #             idx += 1

    #     # ------------ 4. Bottom surface (z = 0) ------------
    #     bottom_mesh = mesh.Mesh(np.zeros(top_tri_count, dtype=mesh.Mesh.dtype))

    #     idx = 0
    #     for i in range(H - 1):
    #         for j in range(W - 1):

    #             i0 = i * aspect_ratio
    #             i1 = (i + 1) * aspect_ratio

    #             # Triangle 1
    #             bottom_mesh.vectors[idx][0] = [i0, j, 0]
    #             bottom_mesh.vectors[idx][1] = [i1, j, 0]
    #             bottom_mesh.vectors[idx][2] = [i0, j+1, 0]
    #             idx += 1

    #             # Triangle 2
    #             bottom_mesh.vectors[idx][0] = [i1, j+1, 0]
    #             bottom_mesh.vectors[idx][1] = [i0, j+1, 0]
    #             bottom_mesh.vectors[idx][2] = [i1, j, 0]
    #             idx += 1

    #     # ------------ 5. Build SIDE walls ------------
    #     side_meshes = []

    #     # Helper to add a wall strip
    #     def add_wall(x1, y1, z1, x2, y2, z2):
    #         """Two points top; bottom is z=0."""
    #         m = mesh.Mesh(np.zeros(2, dtype=mesh.Mesh.dtype))

    #         # Triangle 1
    #         m.vectors[0][0] = [x1, y1, z1]
    #         m.vectors[0][1] = [x1, y1, 0]
    #         m.vectors[0][2] = [x2, y2, z2]

    #         # Triangle 2
    #         m.vectors[1][0] = [x2, y2, z2]
    #         m.vectors[1][1] = [x1, y1, 0]
    #         m.vectors[1][2] = [x2, y2, 0]

    #         return m

    #     # Front wall j = 0
    #     for i in range(H - 1):
    #         z1 = height_map[i, 0]
    #         z2 = height_map[i+1, 0]
    #         side_meshes.append(add_wall(i*aspect_ratio, 0, z1, (i+1)*aspect_ratio, 0, z2))

    #     # Back wall j = W - 1
    #     for i in range(H - 1):
    #         z1 = height_map[i, W-1]
    #         z2 = height_map[i+1, W-1]
    #         side_meshes.append(add_wall(i*aspect_ratio, W-1, z1, (i+1)*aspect_ratio, W-1, z2))

    #     # Left wall i = 0
    #     for j in range(W - 1):
    #         z1 = height_map[0, j]
    #         z2 = height_map[0, j+1]
    #         side_meshes.append(add_wall(0, j, z1, 0, j+1, z2))

    #     # Right wall i = H - 1
    #     for j in range(W - 1):
    #         z1 = height_map[H-1, j]
    #         z2 = height_map[H-1, j+1]
    #         side_meshes.append(add_wall((H-1)*aspect_ratio, j, z1, (H-1)*aspect_ratio, j+1, z2))

    #     # ------------ 6. Combine all meshes ------------
    #     all_meshes = [top_mesh, bottom_mesh] + side_meshes

    #     # Merge into single numpy array
    #     total_triangles = sum(m.vectors.shape[0] for m in all_meshes)
    #     full_mesh = mesh.Mesh(np.zeros(total_triangles, dtype=mesh.Mesh.dtype))

    #     idx = 0
    #     for m in all_meshes:
    #         tri = m.vectors.shape[0]
    #         full_mesh.vectors[idx:idx+tri] = m.vectors
    #         idx += tri

    #     # ------------ 7. Save STL ------------
    #     base_name = os.path.splitext(os.path.basename(image_path))[0]
    #     output_filename = f"{base_name}_solid.stl"
    #     output_path = os.path.join("static/models", output_filename)
    #     full_mesh.save(output_path)

    #     return output_path


class DepthTo3DService:
    """Service to convert depth maps to 3D STL models"""
    
    @staticmethod
    def convert_depth_to_stl(
        image_data: bytes,
        depth_div_width: float,
        output_path: str,
        aspect_ratio: float = 1.0
    ) -> str:
        """
        Convert a depth map image to an STL 3D model
        
        Args:
            image_data: Raw image bytes
            depth_div_width: Ratio of desired depth to width (e.g., 0.5)
            output_path: Path where to save the STL file
            aspect_ratio: Height to width ratio for the model (default: 1.0)
            
        Returns:
            Path to the generated STL file
        """
        # Decode image from bytes
        nparr = np.frombuffer(image_data, np.uint8)
        im = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if im is None:
            raise ValueError("Failed to decode image")
        
        # Process image array
        im_array = np.array(im)
        im_array = np.rot90(im_array, -1, (0, 1))
        
        mesh_size = [im_array.shape[0], im_array.shape[1]]
        mesh_max = np.max(im_array)
        
        if mesh_max == 0:
            raise ValueError("Image contains no depth information (all pixels are black)")
        
        # Scale mesh based on depth information
        if len(im_array.shape) == 3:
            # Color image - use first channel
            scaled_mesh = mesh_size[0] * depth_div_width * im_array[:, :, 0] / mesh_max
        else:
            # Grayscale image
            scaled_mesh = mesh_size[0] * depth_div_width * im_array / mesh_max
        
        # Create mesh
        mesh_shape = mesh.Mesh(
            np.zeros((mesh_size[0] - 1) * (mesh_size[1] - 1) * 2, dtype=mesh.Mesh.dtype)
        )
        
        # Generate triangles for the mesh
        for i in range(0, mesh_size[0] - 1):
            for j in range(0, mesh_size[1] - 1):
                mesh_num = i * (mesh_size[1] - 1) + j
                
                # Apply aspect ratio to i coordinate (height)
                i_scaled = i * aspect_ratio
                i1_scaled = (i + 1) * aspect_ratio
                
                # First triangle
                mesh_shape.vectors[2 * mesh_num][2] = [i_scaled, j, scaled_mesh[i, j]]
                mesh_shape.vectors[2 * mesh_num][1] = [i_scaled, j + 1, scaled_mesh[i, j + 1]]
                mesh_shape.vectors[2 * mesh_num][0] = [i1_scaled, j, scaled_mesh[i + 1, j]]
                
                # Second triangle
                mesh_shape.vectors[2 * mesh_num + 1][0] = [i1_scaled, j + 1, scaled_mesh[i + 1, j + 1]]
                mesh_shape.vectors[2 * mesh_num + 1][1] = [i_scaled, j + 1, scaled_mesh[i, j + 1]]
                mesh_shape.vectors[2 * mesh_num + 1][2] = [i1_scaled, j, scaled_mesh[i + 1, j]]
        
        # Save mesh to file
        mesh_shape.save(output_path)
        return output_path