#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>


namespace {
namespace local {

void cube()
{
	typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

	MyMesh mesh;

	// Generate vertices.

	MyMesh::VertexHandle vhandle[8];

	vhandle[0] = mesh.add_vertex(MyMesh::Point(-1, -1,  1));
	vhandle[1] = mesh.add_vertex(MyMesh::Point( 1, -1,  1));
	vhandle[2] = mesh.add_vertex(MyMesh::Point( 1,  1,  1));
	vhandle[3] = mesh.add_vertex(MyMesh::Point(-1,  1,  1));
	vhandle[4] = mesh.add_vertex(MyMesh::Point(-1, -1, -1));
	vhandle[5] = mesh.add_vertex(MyMesh::Point( 1, -1, -1));
	vhandle[6] = mesh.add_vertex(MyMesh::Point( 1,  1, -1));
	vhandle[7] = mesh.add_vertex(MyMesh::Point(-1,  1, -1));


	// Generate (quadrilateral) faces.

	std::vector<MyMesh::VertexHandle>  face_vhandles;

	face_vhandles.clear();
	face_vhandles.push_back(vhandle[0]);
	face_vhandles.push_back(vhandle[1]);
	face_vhandles.push_back(vhandle[2]);
	face_vhandles.push_back(vhandle[3]);
	mesh.add_face(face_vhandles);

	face_vhandles.clear();
	face_vhandles.push_back(vhandle[7]);
	face_vhandles.push_back(vhandle[6]);
	face_vhandles.push_back(vhandle[5]);
	face_vhandles.push_back(vhandle[4]);
	mesh.add_face(face_vhandles);

	face_vhandles.clear();
	face_vhandles.push_back(vhandle[1]);
	face_vhandles.push_back(vhandle[0]);
	face_vhandles.push_back(vhandle[4]);
	face_vhandles.push_back(vhandle[5]);
	mesh.add_face(face_vhandles);

	face_vhandles.clear();
	face_vhandles.push_back(vhandle[2]);
	face_vhandles.push_back(vhandle[1]);
	face_vhandles.push_back(vhandle[5]);
	face_vhandles.push_back(vhandle[6]);
	mesh.add_face(face_vhandles);

	face_vhandles.clear();
	face_vhandles.push_back(vhandle[3]);
	face_vhandles.push_back(vhandle[2]);
	face_vhandles.push_back(vhandle[6]);
	face_vhandles.push_back(vhandle[7]);
	mesh.add_face(face_vhandles);

	face_vhandles.clear();
	face_vhandles.push_back(vhandle[0]);
	face_vhandles.push_back(vhandle[3]);
	face_vhandles.push_back(vhandle[7]);
	face_vhandles.push_back(vhandle[4]);
	mesh.add_face(face_vhandles);


	// Write mesh to output.obj.
	if (!OpenMesh::IO::write_mesh(mesh, "./data/geometry/openmesh/output.off") )
		std::cerr << "Cannot write mesh to file 'output.off'" << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_openmesh {

void simple_object()
{
	local::cube();
}

}  // namespace my_openmesh
