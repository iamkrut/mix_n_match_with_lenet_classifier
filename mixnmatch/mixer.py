import os
import vtk
from random import randint
from matplotlib import pyplot as plt
from vtk.util import numpy_support

from mesh_parser import *
from utilities import *

def generate(chair_no):

    chair_dirs = [f.path for f in os.scandir("Dataset") if f.is_dir() ]   

    backs = []
    seats = []
    legs = []

    for chair_dir in chair_dirs:
        backs.append(MeshParser(os.path.join(chair_dir, "back.obj"), os.path.join(chair_dir, "chair.obj")))
        seats.append(MeshParser(os.path.join(chair_dir, "seat.obj"), os.path.join(chair_dir, "chair.obj")))

        all_legs = []
        all_legs.append(MeshParser(os.path.join(chair_dir, "leg_front_right.obj"), os.path.join(chair_dir, "chair.obj")))
        all_legs.append(MeshParser(os.path.join(chair_dir, "leg_front_left.obj"), os.path.join(chair_dir, "chair.obj")))
        all_legs.append(MeshParser(os.path.join(chair_dir, "leg_back_left.obj"), os.path.join(chair_dir, "chair.obj")))
        all_legs.append(MeshParser(os.path.join(chair_dir, "leg_back_right.obj"), os.path.join(chair_dir, "chair.obj")))

        legs.append(all_legs)

    print("Types of backs: ", len(backs))
    print("Types of seats: ",len(seats))
    print("Types of legs: ",len(legs))
    chair_count = len(backs) - 1

    meshes = []
    back_idx = randint(0, chair_count)
    meshes.append(backs[back_idx])

    # Seat
    seat_idx = randint(0, chair_count)
    meshes.append(seats[seat_idx])
    # Reference seat parts
    ref_seat_back = backs[seat_idx]
    ref_seat_legs = []
    for i in range(4):
        ref_seat_legs.append(legs[seat_idx][i])

    leg_idx = randint(0, chair_count)
    for i in range(4):
        meshes.append(legs[leg_idx][i])

    # Combine the meshes polydata into different actors
    mesh_actors = append_meshes(meshes)
    bounding_box_actors = append_bounding_boxes(meshes)

    ## scaling the legs to equal scale ##
    # minZScale = 100
    # for i in range(4):
    #     leg_mesh_scale = meshes[2+i].get_scale()
    #     if(leg_mesh_scale[2] < minZScale):
    #         minZScale = leg_mesh_scale[2]

    # for i in range(4):
    #     refZscale = meshes[2+i].get_scale()[2]
    #     scale_to = refZscale/minZScale
    #     mesh_actors[0].SetScale(1, 1, scale_to)

    #### Figuring out legs ####
    ## Translation ##
    # for i in range(len(all_legs)):    
    #     # ref_center_offset = ref_seat_legs[i].get_top_center_point() - ref_seat_legs[i].get_center_point()
    #     # center_offset = meshes[2 + i].get_top_center_point() - meshes[2 + i].get_center_point()
    #     translate_by = ref_seat_legs[i].get_top_center_point() - meshes[2 + i].get_center_point()
    #     # print("Center moved from: ", mesh_actors[2 + i].GetCenter())
    #     mesh_actors[2 + i].SetPosition(translate_by[0], translate_by[1], translate_by[2])
    #     # print("Center moved to: ", mesh_actors[2 + i].GetCenter())
    
    ## Fix the leg connection using voxel collision
    leg_to_seat_connections = [False, False, False, False]
    count = 1

    leg = MeshParser(poly_data = transformPolyData(mesh_actors[2+i]), scale = meshes[2+i].get_relative_scale(), translation = meshes[2+i].get_relative_translation())
    seat = MeshParser(poly_data = transformPolyData(mesh_actors[1]), scale = meshes[1].get_relative_scale(), translation = meshes[0].get_relative_translation())

    origial_leg_center = []
    origial_leg_top_center = []
    
    # # translate legs until collision
    while(not (leg_to_seat_connections[0] and leg_to_seat_connections[1] and leg_to_seat_connections[2] and leg_to_seat_connections[3])):
        
        # check for removing collision errors due to voxel
        if(leg_to_seat_connections[0] == True or leg_to_seat_connections[1] == True):
            leg_to_seat_connections[0] = True
            leg_to_seat_connections[1] = True
        
        if(leg_to_seat_connections[2] == True or leg_to_seat_connections[3] == True):
            leg_to_seat_connections[2] = True
            leg_to_seat_connections[3] = True

        if (leg_to_seat_connections[0] and leg_to_seat_connections[1] and leg_to_seat_connections[2] and leg_to_seat_connections[3]):
            break

        offset = 0.02
        for i in range (4):

            leg = MeshParser(poly_data = transformPolyData(mesh_actors[2+i]), scale = meshes[2+i].get_relative_scale(), translation = meshes[2+i].get_relative_translation())
            seat = MeshParser(poly_data = transformPolyData(mesh_actors[1]), scale = meshes[1].get_relative_scale(), translation = meshes[1].get_relative_translation())

            if (count == 1):
                origial_leg_center.append(leg.get_center_point())
                origial_leg_top_center.append(leg.get_top_center_point() - leg.get_center_point())   
            
            if(not leg_to_seat_connections[i]):
            
                if(not (leg_to_seat_connections[0] or leg_to_seat_connections[1] or leg_to_seat_connections[2] or leg_to_seat_connections[3])):
                    zStep = count * 0.07
                xStep = count * 0.02
                yStep = count * 0.02
                if(i == 0):
                    seat_offset = seat.get_center_front_right_corner() - origial_leg_center[i]
                    mesh_actors[2+i].SetPosition(seat_offset[0] + xStep - offset, seat_offset[1] + yStep - offset, seat_offset[2] + origial_leg_center[i][2] + zStep - offset)
                    temp = [seat_offset[0] + count * 0.01 - offset, seat_offset[1] + count * 0.01 - offset, seat_offset[2] + origial_leg_center[i][2] + zStep - offset]
                    print("leg 0 new position: ", temp)
                elif(i == 1):
                    seat_offset = seat.get_center_front_left_corner() - origial_leg_center[i]
                    mesh_actors[2+i].SetPosition(seat_offset[0] - xStep + offset, seat_offset[1] + yStep - offset, seat_offset[2] + origial_leg_center[i][2] + zStep - offset)
                    temp = [seat_offset[0] - xStep + offset, seat_offset[1] + yStep - offset, seat_offset[2] + origial_leg_center[i][2] + zStep - offset]
                    print("leg 1 new position: ", temp)
                elif(i == 2):
                    seat_offset = seat.get_center_back_left_corner()  - origial_leg_center[i]
                    mesh_actors[2+i].SetPosition(seat_offset[0] - xStep + offset, seat_offset[1] - yStep + offset, seat_offset[2] + origial_leg_center[i][2] + zStep - offset)
                    temp = [seat_offset[0] - xStep + offset, seat_offset[1] - yStep + offset, seat_offset[2] + origial_leg_center[i][2] + zStep - offset]
                    print("leg 2 new position: ", temp)
                elif(i == 3):
                    seat_offset = seat.get_center_back_right_corner() - origial_leg_center[i]
                    mesh_actors[2+i].SetPosition(seat_offset[0] + xStep - offset, seat_offset[1] - yStep + offset, seat_offset[2] + origial_leg_center[i][2] + zStep - offset)
                    temp = [seat_offset[0] + xStep - offset, seat_offset[1] - yStep + offset, seat_offset[2] + origial_leg_center[i][2] + zStep - offset]
                    print("leg 3 new position: ", temp)
            
                # check collision
                if(not leg_to_seat_connections[i]):
                    leg = MeshParser(poly_data = transformPolyData(mesh_actors[2+i]), scale = meshes[2+i].get_relative_scale(), translation = meshes[2+i].get_relative_translation())
                    leg_to_seat_connections[i] = is_intersect(leg, seat)
                    print("leg ", i, " and seat intersect: ", leg_to_seat_connections[i])

        count = count + 1
        if (count == 15):
            print("Stopping leg iteration due to count out")
            break

    #### Figuring out back ####
    ## Scale ##
    ref_seat_back_scale = ref_seat_back.get_scale()
    back_scale = meshes[0].get_scale()
    seat_scale = meshes[1].get_scale()

    # print (np.divide(ref_seat_back_scale, back_scale))
    
    # print("Ref Scale: ", ref_seat_back_scale)
    # print("Scale: ", back_scale)
    
    #print("Scale before: ", mesh_actors[0].GetScale())
    if (np.divide(ref_seat_back_scale, back_scale)[0] < np.divide(seat_scale, back_scale)[0]):
        scale_to = np.divide(ref_seat_back_scale, back_scale)
        mesh_actors[0].SetScale(scale_to[0], 1, 1)
    else:
        scale_to = np.divide(seat_scale, back_scale)
        mesh_actors[0].SetScale(scale_to[0], 1, 1)
    # print("Scale after: ", mesh_actors[0].GetScale())
    
    ## Fix the back connection using voxel collision
    back_to_seat_connection = False
    count = 1
    origial_back_bottom_center = [0,0,0]

    zOffset = 0.05
    yOffset = 0.05
    # translate back until collision
    while(not back_to_seat_connection):

        back = MeshParser(poly_data = transformPolyData(mesh_actors[0]), scale = meshes[0].get_relative_scale(), translation = meshes[0].get_relative_translation())
        seat = MeshParser(poly_data = transformPolyData(mesh_actors[1]), scale = meshes[1].get_relative_scale(), translation = meshes[1].get_relative_translation())
        
        if (count == 1):
            origial_back_center = back.get_center_point()

        if(not back_to_seat_connection):
            zStep = count * 0.015
            yStep = count * 0.015

            translate_by = seat.get_center_back_bottom_point() - origial_back_center
            mesh_actors[0].SetPosition(translate_by[0], translate_by[1] + yOffset - yStep, translate_by[2] + origial_back_center[2] - zStep + zOffset)
            print("Back new position: ", [translate_by[0], translate_by[1] + yOffset - yStep, translate_by[2] + origial_back_center[2] - zStep + zOffset])

            # check collision of back with seat
            back = MeshParser(poly_data = transformPolyData(mesh_actors[0]), scale = meshes[0].get_relative_scale(), translation = meshes[0].get_relative_translation())
            back_to_seat_connection = is_intersect(back, seat)
            print("Back and seat intersect: ", back_to_seat_connection)

        count = count + 1
        if (count == 10):
            print("Stopping back iteration due to count out")
            # count = 1
            # zOffset = zOffset + 0.1
            # yOffset = yOffset + 0.05
            break

    # voxalize and save projection

    # Combine the meshes polydata into one mesh polydata
    combined_polydata = combine_meshes_polydata(mesh_actors)

    #rotate the combined Mesh to 180 on z axis
    chair_mesh = MeshParser(poly_data=combined_polydata)
    chair_mesh.set_is_normalized()
    chair_mesh.rotate(-90, 0)
    [top, left, front] = chair_mesh.get_projections(512)

    # # Write the stl file to disk
    # objWriter = vtk.vtkSTLWriter()
    # objWriter.SetFileName(os.path.join('Generated_Meshes','{}.stl'.format(chair_no)))
    # objWriter.SetInputConnection(combined_polydata.GetClassName())
    # print("Saving the mesh")
    # objWriter.Write()

    save_projection(top, left, front, chair_no)    

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the mesh actors to the scene
    for actor in mesh_actors:
        actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        renderer.AddActor(actor)
    
    # Add the boundingbox actors to the scene
    # for actor in bounding_box_actors:
    #     actor.GetProperty().SetColor(0, 0, 0)
    #     renderer.AddActor(actor)

    renderer.SetBackground(1.0, 1.0, 1.0) #  Background color dark red

    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()

