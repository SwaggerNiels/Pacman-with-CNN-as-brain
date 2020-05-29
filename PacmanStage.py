import sys
import random
import numpy as np;
import math;
import time;
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Canvas, ALL, NW;
import neuralnetwork as nn;

class Cons:

    BOARD_WIDTH = 1200;
    BOARD_HEIGHT = 800;
    DELAY = 1048;
    MAX_DELAY = 2048;
    U = 40;
    
    PAC_VIEW_X = 40;
    PAC_VIEW_Y = 600;
    PAC_VIEW_W = 180;
    PAC_VIEW_H = 180;
    
    GLOBALS_X = 1010;
    GLOBALS_Y = 10;
    

class Board(Canvas):

    def __init__(self, game_mode = "field"): 
        super().__init__(width=Cons.BOARD_WIDTH, height=Cons.BOARD_HEIGHT,
            background="black", highlightthickness=0)
        
        self.time = 0;
        self.noise = 100;
        self.game_mode = game_mode;
        self.game_mode_store = None;
        
        if game_mode == "pause":
            self.initDataCollection();
            self.initGame();
        
        self.pack();

    def initGame(self):
        '''initializes game'''

        self.score = 0;
        Cons.DELAY = 128;

        # variables used to pacman
        self.moving = False;
        self.dir = "Right";
        self.speed = Cons.U*1;
        
        self.dir_dictX = {
            "Right" : 1,
            "Left" : -1,
            "Up" : 0,
            "Down" : 0,
            };
        
        self.dir_dictY = {
            "Up" : -1,
            "Down" : 1,
            "Right" : 0,
            "Left" : 0,
            };
        
        self.dir_opposite_dict = {
            "Right" : "Left",
            "Left" : "Right",
            "Up" : "Down",
            "Down" : "Up",
            };
        
        #pacman view
        self.spot_length = Cons.U*5;
        self.spot_width = Cons.U*.8;
        self.i_name_blank = "blank.png";
        self.spot_i_name = self.i_name_blank;
        
        #pacman thoughts
        self.previous_move = (0,0);     
        self.pacman_thoughts = "nothing ahead...";
        self.pacman_memory = [""]*5;

        # starting ghost coordinates
        self.ghosts = [];
        self.i_name_ghosts = [];
        self.ghost_speed = Cons.U*.25;
        self.ghostX = 400;
        self.ghostY = 200;
        
        # starting friend coordinates
        self.friends = [];
        self.i_name_friends = [];
        self.friendX = 600;
        self.friendY = 200;
        
        # walls
        self.walls = [];
        self.i_name_walls = [];
        
        self.loadImages();

        self.stats = ["game_mode_text","score_text","time_text",
                      "speed_text","noise_text","items_text", "key_text"];
        self.createObjects();
        
        self.getting_walls = True;
        
        self.key_last_pressed = None;
        self.bind_all("<Key>", self.onKeyPressed);
        
        self.m_x = None;
        self.m_y = None;
        self.bind("<Button 1>", self.onMouseLeft);
        self.bind("<Button 2>", self.onMouseMiddle);
        self.bind("<Button 3>", self.onMouseRight);
        self.after(Cons.DELAY, self.onTimer);
        
        #for lab gamemode
        self.angle = 0;
        
        pass;
    
    def initDataCollection(self):
        '''starts the data collection mode'''
        
        self.pacman_spawnX = 200;
        self.pacman_spawnY = 200;
        self.spawnX = 200;
        self.spawnY = 200;
        
        self.neuralnet_dict = {
            0 : "Nothing...",
            1 : "Ghost, run!",
            2 : "Coin :)",
            3 : "Wall far away",
            4 : "Wall in my face",
            };
        
        self.spawn_dict = {
            0 : "nothing",
            1 : "ghost",
            2 : "friend",
            3 : "wall",
            };
        
        self.brain_input_shape = (100*4,1);
        self.compression = Cons.U*Cons.U//self.brain_input_shape[0];
        self.brain_shape = (self.brain_input_shape[0],20,20,len(self.neuralnet_dict));
        self.pacman_brain = nn.CNN(self.brain_shape[1:], 
                                   image_dimension = 20, 
                                   kernel_size = 3, 
                                   kernel_number = 4, 
                                   maxpool_dimension = 10);
        
        self.pacman_brain.setLayerKernels();
        
        #set the spawning for the data collection
        self.number_spawned = 0;
        self.spawn_number = 1000;
        self.spawn_sequence = np.random.randint(len(self.spawn_dict),size = (self.spawn_number));
        self.spawn_distance_sequence = np.random.randint(1, 6,size = (self.spawn_number));
        self.spawn_distance_sequence[self.spawn_sequence == 0] = 0;
        self.spawn_subject = None;
        
        self.brain_inputs = np.zeros((len(self.spawn_sequence),self.brain_input_shape[0],1));
        self.brain_labels = np.zeros((len(self.spawn_sequence),len(self.neuralnet_dict),1));
        for i in range(len(self.spawn_sequence)):
            #distinct case for close wall
            if self.spawn_distance_sequence[i] == 1 and self.spawn_sequence[i] == 3:
                self.brain_labels[i,4] = 1;
            else:
                self.brain_labels[i,self.spawn_sequence[i]] = 1;
        pass;
        
    def loadImages(self):
        '''loads images from the disk'''
        
        try:
            self.ipacman = dict();
            self.pacman = dict();
            
            image_name = "pacman_l.png";
            self.ipacman["Left"] = Image.open(image_name);
            self.pacman["Left"] = ImageTk.PhotoImage(self.ipacman["Left"]);
            image_name = "pacman_r.png";
            self.ipacman["Right"] = Image.open(image_name);
            self.pacman["Right"] = ImageTk.PhotoImage(self.ipacman["Right"]);
            image_name = "pacman_u.png";
            self.ipacman["Up"] = Image.open(image_name);
            self.pacman["Up"] = ImageTk.PhotoImage(self.ipacman["Up"]);
            image_name = "pacman_d.png";
            self.ipacman["Down"] = Image.open(image_name);
            self.pacman["Down"] = ImageTk.PhotoImage(self.ipacman["Down"]);
            
            #add ghost image
            image_name = "ghost.png";
            self.ighost= Image.open(image_name);
            self.ghost = ImageTk.PhotoImage(self.ighost);
            self.i_name_ghosts.append(image_name);
            
            #add friend image
            image_name = "friend.png";
            self.ifriend = Image.open(image_name);
            self.friend = ImageTk.PhotoImage(self.ifriend);
            self.i_name_friends.append(image_name);
            
            #add wall image
            image_name = "wall.png";
            self.iwall = Image.open(image_name);
            self.wall = ImageTk.PhotoImage(self.iwall);
            self.i_name_walls.append(image_name);
            
            #pacman view
            image = Image.open(self.spot_i_name).resize((100,100));
            self.subject_image = ImageTk.PhotoImage(image);
            
            #background default
            image = Image.open("background_default.png");
            self.bg_default = ImageTk.PhotoImage(image);
            
            #background lab
            image = Image.open("background_lab.png");
            self.bg_lab = ImageTk.PhotoImage(image);
            
        except IOError as e:
            
            print(e);
            sys.exit(1);

    def createObjects(self):
        '''creates room objects on Canvas'''
        
        #background
        self.create_image(0, 0, image=self.bg_default, anchor=NW,  tag="background");
        
        #pacman
        px = 200;
        py = 200;
        
        self.create_image(px, py, image=self.pacman["Right"], anchor=NW,  tag="pacman");
        
        #place spot rectangle for pacman
        pxc = px + Cons.U*.5;
        pyc = py + Cons.U*.5;
        
        lri = self.dir_dictX[self.dir];
        udi = self.dir_dictY[self.dir];
            
        r1 = (pxc + self.spot_width/2*udi ,
              pyc + self.spot_width/2*lri);
        r2 = (pxc + self.spot_length*lri - self.spot_width/2*udi,
              pyc + self.spot_length*udi - self.spot_width/2*lri);
        
        self.create_rectangle((r1[0], r1[1], r2[0], r2[1]), outline = "blue", tag = "spot_line");
                
        
        bbox = np.array([
            (Cons.PAC_VIEW_X+(Cons.PAC_VIEW_W+30)*(i),
             Cons.PAC_VIEW_Y+1,
             Cons.PAC_VIEW_X+(Cons.PAC_VIEW_W+30)*(i+1),
             Cons.PAC_VIEW_Y+Cons.PAC_VIEW_H) for i in range(5)]);
        
        rect_adjust = np.array([0,0,-20,0]);
        
        #pacman view
        self.create_rectangle(tuple(bbox[0]+rect_adjust), tag="pacman_view", outline="white")
        self.create_text((bbox[0][0] + bbox[0][2])*.5, bbox[0][3]-10, text="Pacman view",
                         tag="pacman_view_tag", fill="white")
        self.create_image(bbox[0][0] + 18, bbox[0][1] + 5, anchor = NW, image=self.subject_image, tag="view");
        
        #pacman thoughts
        self.create_rectangle(tuple(bbox[3]+rect_adjust), tag="pacman_view", outline="white")
        self.create_text((bbox[3][0] + bbox[3][2])*.5, bbox[3][1]+40, text=self.pacman_thoughts,
                         tag="pacman_thoughts", fill="white", font=("Calibri", 20));
        self.create_text((bbox[3][0] + bbox[3][2])*.5, bbox[3][3]-10, text="Pacman's thoughts",
                         tag="pacman_thoughts_tag", fill="white");
        
        #pacman memory
        self.create_rectangle(tuple(bbox[4]+rect_adjust), tag="pacman_view", outline="white")
        self.create_text((bbox[4][0] + bbox[4][2])*.5, bbox[2][1]+60, text="\n".join(self.pacman_memory),
                         tag="pacman_memory", fill="white", font=("Calibri", 12));
        self.create_text((bbox[4][0] + bbox[4][2])*.5, bbox[2][3]-10, text="Pacman's memory",
                         tag="pacman_memory_tag", fill="white");
        
        #pacman brain
        bbox_brain = (bbox[1][0],bbox[1][1],bbox[2][2],bbox[2][3]);
        self.create_rectangle(bbox_brain, tag="pacman_brain", outline="white")
        self.create_text((bbox_brain[0] + bbox_brain[2])*.5, bbox_brain[3]-10, text="Pacman's brain \
                         (partial representation of actual CNN)",
                         tag="pacman_brain_tag", fill="white");
        self.bbox_brain = bbox_brain;
        
        
        x = Cons.GLOBALS_X;
        y = Cons.GLOBALS_Y;
        
        for stat in self.stats:
            self.create_text(x, y, tag=stat, fill="white", anchor=NW);
            y += 20;

    # Subjects

    def moveGhosts(self):
        
        pacman = self.find_withtag("pacman");
        px,py = self.coords(pacman);
        x_not_y = np.random.rand() > .5;
        
        for ghost in self.ghosts:
            x,y = self.coords(ghost);
            
            mx = 0;
            my = 0;
            if np.sign(px-x) and np.sign(py-y):
                if x_not_y:
                    mx = np.sign(px-x)*self.ghost_speed;
                else:
                    my = np.sign(py-y)*self.ghost_speed;
            elif np.sign(px-x):
                mx = np.sign(px-x)*self.ghost_speed;
            elif np.sign(py-y):
                my = np.sign(py-y)*self.ghost_speed;
            
            self.move(ghost,mx,my);
            
    def createSubject(self, subject_kind, x, y):
        
        if subject_kind == "ghost":
            subject_image = self.ghost;
            subject = self.create_image(x, y, image=subject_image,
                                        anchor=NW, tag=subject_kind);
            self.ghosts.append(subject);
        elif subject_kind == "friend":
            subject_image = self.friend;
            subject = self.create_image(x, y, image=subject_image,
                                        anchor=NW, tag=subject_kind);
            self.friends.append(subject); 
        elif subject_kind == "wall":
            subject_image = self.wall;
            subject = self.create_image(x, y, image=subject_image,
                                        anchor=NW, tag=subject_kind);
            self.walls.append(subject); 
        
        return(subject);
    
    def removeSubject(self, subject_kind, subject = None):
        
        if subject == None:
            subjects = self.find_withtag(subject_kind);
            for subject in subjects:
                self.delete(subject);
                
                if subject_kind == "ghost":
                    self.ghosts.remove(subject);
                elif subject_kind == "friend":
                    self.friends.remove(subject);
                elif subject_kind == "wall":
                    self.walls.remove(subject);
            
        else:
            self.delete(subject);
        
            if subject_kind == "ghost":
                self.ghosts.remove(subject);
            elif subject_kind == "friend":
                self.friends.remove(subject);
            elif subject_kind == "wall":
                self.walls.remove(subject);
        
    def presentSubjects(self):
        
        f = lambda a : (2*math.sin(a),4*math.cos(a));
        
        self.angle += 10/360;
        self.angle %= 360;
        
        
        if len(self.walls) == 0:
            wall = self.createSubject("wall", 350, 320);
        else:
            wall = self.find_withtag("wall")[0];
        
        mx,my = f(self.angle+90);
        self.move(wall,mx,my);
        
        if len(self.friends) == 0:
            friend = self.createSubject("friend", 350, 80);
        else:
            friend = self.find_withtag("friend")[0];
        
        mx,my = f(self.angle-90);
        self.move(friend,mx,my);
        
        if len(self.ghosts) == 0:
            ghost = self.createSubject("ghost", 250, 200);
        else:
            ghost = self.find_withtag("ghost")[0];
        
        mx,my = f(self.angle);
        self.move(ghost,mx,my);
    
    def getWalls(self,name="walls_room"):
        
        n = Cons.BOARD_WIDTH//Cons.U;
        m = Cons.BOARD_HEIGHT//Cons.U;
        
        wall_matrix = np.zeros((m,n));
        
        for i in range(n):
            for j in range(m):
                if self.checkCollisions(i*Cons.U, j*Cons.U) != None:
                    wall_matrix[j,i] = ("wall" in self.checkCollisions(i*Cons.U, j*Cons.U));
        
        print(f"Saved the walls in the room to name: '{name}'");
        
        np.save(name,wall_matrix);
    
    def setWalls(self,name="walls_room"):
        
        n = Cons.BOARD_WIDTH//Cons.U;
        m = Cons.BOARD_HEIGHT//Cons.U;
        
        wall_matrix = np.load(name, allow_pickle = True);
        
        for i in range(n):
            for j in range(m):
                if wall_matrix[j,i]:
                    x,y = (i*Cons.U, j*Cons.U);
                    self.createSubject("wall",x,y);
                    
        print(f"Loaded the walls of scene: '{name}'");
        
        np.save(name,wall_matrix);
    
    # Pacman

    def checkSubjectCollision(self, subject_kind = "ghost"):
        '''checks if pacman collides with ghost'''
        
        if subject_kind == "ghost" and len(self.ghosts) != 0:
            ghost = self.find_withtag("ghost")
            pacman = self.find_withtag("pacman")
            
            x1, y1, x2, y2 = self.bbox(pacman)
            overlap = self.find_overlapping(x1, y1, x2, y2);
            
            for ovr in overlap:
                if ghost[0] == ovr:
                    self.score -= 1;
                    x, y = self.coords(ghost);
                    
                    if self.game_mode == "field":
                        self.locateSubject("ghost");
        elif subject_kind == "friend" and len(self.friends) != 0:
            friend = self.find_withtag("friend")
            pacman = self.find_withtag("pacman")
            
            x1, y1, x2, y2 = self.bbox(pacman)
            overlap = self.find_overlapping(x1, y1, x2, y2);
            
            for ovr in overlap:
                if friend[0] == ovr:
                    print("collided with coin")
                    self.score += 1;
                    x, y = self.coords(friend);
                    
                    if self.game_mode == "field":
                        self.locateSubject("friend");
        pass;

    def movePacman(self, mx, my):
        '''moves the pacman object'''
        
        pacman = self.find_withtag("pacman");
        spot_line = self.find_withtag("spot_line");
        
        if self.moving:
            self.move(pacman, mx, my);
            self.move(spot_line, mx, my);

    def checkCollisions(self, x, y, subject = "all", echo = False):
        '''checks for collisions'''
        x1 = x//Cons.U * Cons.U;
        y1 = y//Cons.U * Cons.U;
        
        x2 = (x//Cons.U + 1) * Cons.U;
        y2 = (y//Cons.U + 1) * Cons.U;
        
        overlap = self.find_enclosed(x1,y1,x2,y2);
        
        if echo: print("within this box are: ");
        for ovr in overlap:
            if echo: print(self.gettags(ovr));
            return(self.gettags(ovr));
        
        return(None);
        
    def lookForSubject(self, place = "spot"):
        '''Let pacman look in front of him to spot someone'''
        pacman = self.find_withtag("pacman");
        px, py = self.coords(pacman);
        
        if (len(self.ghosts) != 0 or
            len(self.friends) != 0 or
            len(self.walls) != 0):
            
            if place == "spot":
                bbox = self.find_withtag("spot_line");
                x1, y1, x2, y2 = self.bbox(bbox);
            elif place == "pacman":
                bbox = self.find_withtag("pacman");
                x1, y1, x2, y2 = self.bbox(bbox);
            elif place == "mouse":
                x1, y1, x2, y2 = (self.m_x, self.m_y, self.m_x+Cons.U, self.m_y+Cons.U)
            
            overlap = self.find_overlapping(x1, y1, x2, y2);
        
            s_close = None;
            distance = 99999;
            
            for ovr in overlap:
                #ghost
                if len(self.ghosts) != 0:
                    ghost = self.find_withtag("ghost");
                    if ghost[0] == ovr:
                        sx, sy = self.coords(ghost[0]);
                        new_distance = int(abs(px-sx)//Cons.U + abs(py-sy)//Cons.U);
                        if new_distance < distance:
                                distance = new_distance;
                                s_close = ghost[0];
                
                #friends
                if len(self.friends) != 0:
                    friend = self.find_withtag("friend");
                    if friend[0] == ovr:
                        sx, sy = self.coords(friend[0]);
                        new_distance = int(abs(px-sx)//Cons.U + abs(py-sy)//Cons.U);
                        if new_distance < distance:
                            distance = new_distance;
                            s_close = friend[0];
                
                #walls
                if len(self.walls) != 0:
                    wall = self.find_withtag("wall");
                    for s in wall:
                        if s == ovr:
                            sx, sy = self.coords(s);
                            new_distance = int(abs(px-sx)//Cons.U + abs(py-sy)//Cons.U);
                            if new_distance < distance:
                                distance = new_distance;
                                s_close = s;
            
            if s_close != None:
                return(s_close,distance);
                
        return(None,None);

    def locateSubject(self, subject_kind = "ghost"):
        '''places the ghost object on Canvas'''
        
        if subject_kind == "ghost":
            ghost = self.find_withtag("ghost")
            if len(self.ghosts) > 0:       
                self.ghosts.remove(ghost[0]);
            self.delete(ghost[0]);
            
            ghost = self.create_image(self.ghostX, self.ghostY, anchor=NW,
                image=self.ghost, tag="ghost");
            
            self.ghosts.append(ghost);
        elif subject_kind == "friend":
            print("removing coin")
            friend = self.find_withtag("friend")
            if len(self.friends) > 0:       
                self.friends.remove(friend[0]);
            self.delete(friend[0]);
            
            collision = 1;
            while collision != None:
                x = np.random.randint(0,800)//Cons.U*Cons.U;
                y = np.random.randint(0,600)//Cons.U*Cons.U;
                collision = self.checkCollisions(x,y);
                
            friend = self.create_image(x, y, anchor=NW,
                image=self.friend, tag="friend");
            
            self.friends.append(friend);
    
    def letPacmanSee(self, display = True):
        image_name = self.i_name_blank; #pacman's view (starts blank)
        
        # print("Subjects in room:","ghosts =",self.ghosts,"friends =",self.friends);
        subject_insight,subject_distance = self.lookForSubject();
        # print("subject spotted:",subject_insight," - with distance:",subject_distance);
        
        if subject_insight != None:
           # print("Subject in sight:",subject_insight);
           
           if subject_insight in self.ghosts:
               subject_index = self.ghosts.index(subject_insight);
               
               # print("Subject is ghost with index:",subject_index);
               image_name = self.i_name_ghosts[subject_index];
               
           elif subject_insight in self.friends:
               subject_index = self.friends.index(subject_insight);
               
               # print("Subject is friend with index:",subject_index);
               image_name = self.i_name_friends[subject_index];
               
           elif subject_insight in self.walls:
               subject_index = self.walls.index(subject_insight);
               
               # print("Subject is wall with index:",subject_index);
               image_name = self.i_name_walls[0];
        
        self.spot_i_name = image_name;
        pacman_image = self.pacmanProcessVisualStimulus(image_name, subject_distance);

        #change pacman view display (subject_image is the resized image seen on the screen)     
        if display:
            image = pacman_image.resize((Cons.PAC_VIEW_W-25,Cons.PAC_VIEW_H-25));
            self.subject_image = ImageTk.PhotoImage(image);
            self.itemconfigure("view", image = self.subject_image);
        
        return(pacman_image);
        
    def pacmanProcessVisualStimulus(self, image_name, subject_distance, compression = 4):
        '''Process the image of given name, that is observed by pacman. Compress information and add noise'''
        
        pacman_image = Image.open(image_name);
        
        #add distance effect, if distance bigger than 1
        if subject_distance != None and subject_distance > 1:
            shape = np.array(pacman_image.size);
            factor = subject_distance*2;
            small_shape = shape - factor*2;
            small_pacman_image = pacman_image.resize(small_shape);
            
            image_data = np.zeros(list(shape) + [4]);
            image_data[:,:,:3] = 255;
            small_image_data = np.array(small_pacman_image.getdata())
            small_image_data = small_image_data.reshape(list(small_shape) + [4]);

            small_image_noise = np.random.randint(0,self.noise*1.5,small_image_data.shape);
            small_image_data = np.clip(small_image_data+small_image_noise,0,255);
            image_data[factor:-factor,factor:-factor] = small_image_data;
            
            image_data.reshape(list(shape)+[4]);
        else:
            image_data = np.array(pacman_image.getdata());
        #(rows x cols) x 4 (RGBA) np.array of the image: image_data
        
        #compress
        image_data = image_data.reshape(Cons.U,Cons.U,4);
        rows = cols = Cons.U//compression;
        image_data = image_data.reshape(rows, int(image_data.shape[0]/rows), cols, int(image_data.shape[1]/cols), 4).mean(axis=1).mean(axis=2)
        image_data = image_data.reshape(rows*cols,4).astype(int);
        
        #add noise
        image_noise = np.random.randint(0,self.noise,image_data.shape);
        image_data = np.clip(image_data+image_noise,0,255);
        
        #convert back to image
        image_data = tuple(map(tuple,image_data));
        pacman_image = Image.new("RGBA",(rows,cols));
        pacman_image.putdata(image_data);
        
        return(pacman_image);
       
    def pacmanRespondToView(self, display = True):
        
        pacman_image = self.letPacmanSee(display);
        
        if pacman_image != None:
            brain_input = np.array(pacman_image.getdata())[:,:].reshape(self.brain_input_shape)/255;
            
            output_nodes = self.pacman_brain.feedForward(brain_input);
            
            response = np.argmax(output_nodes);
        else:
            reponse = 0;
        
        if display:
            self.pacman_thoughts = self.neuralnet_dict[response];
            
            #update thoughts
            thoughts = self.find_withtag("pacman_thoughts");
            self.itemconfigure(thoughts, text=self.pacman_thoughts);
            
            #update memory
            self.pacman_memory.insert(0,self.pacman_thoughts);
            self.pacman_memory.pop();
            
            #update memory display
            memory = self.find_withtag("pacman_memory");
            self.itemconfigure(memory, text="\n".join((self.pacman_memory)));
        
        return(response);
    
    def actionsOfPacman(self, moving = True):
        
        if self.game_mode == "data collection":
            pacman_image = self.letPacmanSee();
            brain_input = np.array(pacman_image.getdata())[:,:].reshape(self.brain_input_shape)/255;
            
            self.brain_inputs[self.number_spawned,:] = brain_input;
        
        elif self.game_mode == "field" or self.game_mode == "lab":
            
            #move pacman according to its previous decision
            mx,my = self.previous_move;
            self.movePacman(mx,my);
            
            pacman = self.find_withtag("pacman");
            px,py = self.coords(pacman);
            collision = self.lookForSubject(place = "pacman");
            if collision[0] != None and "wall" in self.gettags(collision[0]):
                print("Pacman made a mistake: he thought he could move through a wall...");
                self.moving = True;
                self.movePacman(-mx,-my);
            
            self.moving = False;
            
            response = self.pacmanRespondToView();

            #ghost response
            if response == 1:
                print("flee");
                self.dir = self.dir_opposite_dict[self.dir];
                self.updateSpot();
                self.moving = moving;
            #friend response
            elif response == 2:
                print("get coin");
                self.moving = moving;
            #close wall reponse
            elif response == 4:
                counter = 0;
                
                #save state
                brain_state = np.copy(self.pacman_brain.getActivations());
                while True:
                    self.dir = np.random.choice(["Right","Left","Up","Down"]);
                    self.updateSpot();
                    response = self.pacmanRespondToView(display = False);
                    
                    counter += 1;
                    if counter > 20:
                        print("Pacman is stuck in a corner, he needs more training");
                        break;
                    if response != 4:
                        self.moving = moving;
                        print("wall close, turn")
                        
                        #restore state
                        self.pacman_brain.setActivations(brain_state);
                        break;
            else:
                self.moving = moving;
            
            #move to point
            mx = self.dir_dictX[self.dir]*self.speed;
            my = self.dir_dictY[self.dir]*self.speed;
            
            self.previous_move = (mx,my);
        pass;
    
    def resetPacman(self):
        #reset pacman
        px,py = self.coords(self.find_withtag("pacman"))
        mx = self.pacman_spawnX - px;
        my = self.pacman_spawnY - py;
        
        self.moving = True;
        self.movePacman(mx,my);
        self.moving = False;
        
        self.dir = "Right";
        self.updateSpot();
    
    # Loop functions
    
    def dataCollectionLoop(self):
        
        subjectsInRoom = len(self.ghosts) + len(self.friends) + len(self.walls);
        
        # print("Number of subjects in room:",subjectsInRoom);
        # print("Ghosts in room:",self.ghosts);
        # print("Friends in room:",self.friends);
        
        if self.number_spawned < self.spawn_number:
            
            #despawn previous subject
            if self.spawn_subject != None:
                subject_kind = self.gettags(self.spawn_subject)[0];
                self.removeSubject(subject_kind,self.spawn_subject);
            
            #new spawn (or nothing)
            spawn = self.spawn_dict[self.spawn_sequence[self.number_spawned]];
            
            if spawn != "nothing":
                #spawn subject
                new_subject = spawn;
                seq_x = self.spawn_distance_sequence[self.number_spawned]*Cons.U;
                
                self.spawn_subject = self.createSubject(new_subject, self.spawnX + seq_x, self.spawnY);
            else:
                self.spawn_subject = None;
                
            self.actionsOfPacman(moving = False);
            self.number_spawned += 1;
            
            self.resetPacman();
            
        else:
            if subjectsInRoom == 1:
                #despawn subject
                subject_kind = self.gettags(self.spawn_subject)[0];
                self.removeSubject(subject_kind,self.spawn_subject);
            print("Data collection done")
            
            #train brain with all the inputs
            training_pairs = [(ti,tl) for ti,tl in zip(
                np.array(self.brain_inputs),
                np.array(self.brain_labels))];
            
            self.pacman_brain.printAccuracy(training_pairs);
            
            self.pacman_brain.randomTrainingProcedure(
                training_pairs,
                10,
                1,
                1,
                # training_pairs[:10]
                )
            
            print("Pacman has learned!")
            
            self.drawNeuralNetwork();
            
            self.game_mode = "pause";
            Cons.DELAY = 128;
            self.noise = 100;
            
    def updateSpot(self):
        spot_line = self.find_withtag("spot_line");
        self.delete(spot_line[0]);
        
        pacman = self.find_withtag("pacman");
        px,py = self.coords(pacman);
        
        #place spot rectangle for pacman
        pxc = px + Cons.U*.5;
        pyc = py + Cons.U*.5;
        
        lri = self.dir_dictX[self.dir];
        udi = self.dir_dictY[self.dir];
            
        r1 = (pxc + self.spot_width/2*udi ,
              pyc + self.spot_width/2*lri);
        r2 = (pxc + self.spot_length*lri - self.spot_width/2*udi,
              pyc + self.spot_length*udi - self.spot_width/2*lri);
        
        self.create_rectangle((r1[0], r1[1], r2[0], r2[1]), outline = "blue", tag = "spot_line");
        
        self.itemconfigure(pacman, image = self.pacman[self.dir]);
    
    def onMouseLeft(self, e):
        mx,my = e.x//Cons.U*Cons.U, e.y//Cons.U*Cons.U;
        self.m_x, self.m_y = (mx,my);
        
        if self.game_mode == "pause":
            print("wall created");
            self.createSubject("wall", mx, my);
    
    def onMouseRight(self, e):
        mx,my = e.x//Cons.U*Cons.U, e.y//Cons.U*Cons.U;
        self.m_x, self.m_y = (mx,my);
        
        if self.game_mode == "pause":
            if "wall" in self.checkCollisions(mx,my):
                wall,_ = self.lookForSubject("mouse");
                self.removeSubject("wall", wall);
                print("wall removed");
            
    def onMouseMiddle(self, e):
        mx,my = e.x,e.y;
        self.mr_x, self.mr_y = (mx,my);
        
        self.checkCollisions(mx,my, echo=True);

    def onKeyPressed(self, e):
        '''controls direction variables with cursor keys'''
        key = e.keysym;
        
        self.key_last_pressed = key;
        
        #game speed control
        if key == "w":
            Cons.DELAY = min(Cons.MAX_DELAY,int(Cons.DELAY*2));
        
        if key == "s":
            Cons.DELAY = max(1,int(Cons.DELAY//2));
        
        #game noise control
        if key == "q":
            self.noise = min(255,self.noise+50);
        
        if key == "a":
            self.noise = max(5,self.noise-50);
        
        if key == "p":
            if self.game_mode == "pause":
                self.game_mode = self.game_mode_store;
                self.game_mode_store = None;
            else:
                self.game_mode_store = self.game_mode;
                self.game_mode = "pause";
            
            print("GAMEMODE - ",self.game_mode);
        
        if self.game_mode == "data collection":
            return;
        
        if key == "l":
            if self.game_mode != "lab" and self.game_mode_store == None:
                self.exitGamemode();
                self.resetPacman();
                self.game_mode = "lab";
                self.drawBackground(self.bg_lab);
                print("GAMEMODE - ",self.game_mode);
        
        if key == "d":
            if self.game_mode != "data collection":
                self.exitGamemode();
                self.resetPacman();
                self.game_mode = "data collection";
                print("GAMEMODE - ",self.game_mode);
                print("data!!")
                
        if key == "r":
            self.pacman_brain.resetNetwork();
        
        if key == "o":
            w,b = self.pacman_brain.getNetwork();
            
            print("Output network to file");
            
            np.save("weights",w);
            np.save("biases",b);
                
        if key == "i":
            w = np.load("weights.npy", allow_pickle = True);
            b = np.load("biases.npy", allow_pickle = True);
            
            print("Copy network from file");
            
            self.pacman_brain.setNetwork(w,b);
        
        if key == "f":
            if self.game_mode != "field" and self.game_mode_store == None:
                self.exitGamemode();
                self.game_mode = "field";
                self.createSubject("ghost",self.ghostX,self.ghostY);
                self.createSubject("friend",self.friendX,self.friendY);
            
            print("GAMEMODE - ",self.game_mode);
        
        #when making new scenes and creating presets activate 'm'
        if key == "m":
            self.getting_walls = not self.getting_walls;
            print(f"Getting mode: {self.getting_walls}");
        
        #take preset walls and put in room
        if self.getting_walls:
            if key in list("123456789"):
                self.getWalls("walls_"+key);
                
        #get walls and make new preset
        if not self.getting_walls:
            self.removeSubject("wall");
            if key in list("123456789"):
                self.setWalls(f"walls_{key}.npy");
        
        #motion control of pacman
        keys_motion = ["Left","Right","Up","Down"];
        
        if key == keys_motion[0]:
            
            mx = -Cons.U;
            my = 0;
            self.moving = True;
            
        if key == keys_motion[1]:
            
            mx = Cons.U;
            my = 0;
            self.moving = True;
            
        if key == keys_motion[2]:
            
            mx = 0;
            my = -Cons.U;
            self.moving = True;
            
        if key == keys_motion[3]:
            
            mx = 0;
            my = Cons.U;
            self.moving = True;
        
        
        #set direction and change the spot_line
        if key in keys_motion:
            self.movePacman(mx,my);
            
            old_dir = self.dir;
            self.dir = key;
            
            if old_dir != self.dir:
                self.updateSpot();
        
        e.keysym = "";

    def exitGamemode(self):
        if (self.game_mode == "lab" or
            self.game_mode == "field" or 
            self.game_mode == "data_collection" or
            self.game_mode == "pause"):
            
            self.removeSubject("wall");
            self.removeSubject("friend");
            self.removeSubject("ghost");
            
            self.drawBackground(self.bg_default);
        
        if self.game_mode == "lab":
            self.angle = 0;
        
    def onTimer(self):
        '''creates a game cycle each timer event'''
        
        self.time += 1;
        self.drawGlobals();

        if self.game_mode == "field":
            self.checkSubjectCollision("ghost");
            self.checkSubjectCollision("friend");
            self.actionsOfPacman();
            self.moveGhosts();
            self.after(Cons.DELAY, self.onTimer);
            
        elif self.game_mode == "data collection":
            self.resetPacman();
            self.dataCollectionLoop();
            self.after(Cons.DELAY, self.onTimer);
            
        elif self.game_mode == "lab":
            self.presentSubjects();
            self.actionsOfPacman(moving = False);
            self.resetPacman();
            self.after(Cons.DELAY, self.onTimer);
            
        elif self.game_mode == "pause":
            self.after(Cons.DELAY, self.onTimer);
        else:
            self.gameOver();

        if self.game_mode != "data collection":
            self.drawNeuralNetwork();
        
    # Draw functions
            
    def drawNeuralNetwork(self):
        
        configure_existing = 1 if len(self.find_withtag("nn_draw")) != 0 else 0;
        
        nodes_draw = [min(10,self.pacman_brain.layer_sizes[l]) for l in range(4)];
        
        L = self.pacman_brain.L;
        
        #draw all the weights and nodes for the (non-input) layers
        for l in range(L):
            
            # print('Layer___________',l);
            for i in range(nodes_draw[l]):
                
                if l != 0:
                    activation = self.pacman_brain.getNodeActivation(l, i)[0];
                    
                    width = max(0.01,abs(activation))*5;
                    tag = f"nn_draw{l},{i}";
                    outline = "green";
                    
                    if l == L-1 and i == np.argmax(self.pacman_brain.a[l]):
                        outline = "yellow";
                    
                    if configure_existing:
                        circle = self.find_withtag(tag)[0];
                        self.itemconfigure(circle, width = width, outline = outline);
                    else:
                        bbox = self.bbox_brain;
                        self.create_oval((bbox[0]+10 + l*100,
                                         bbox[1]+10 + 15*i,
                                         bbox[0]+20 + l*100,
                                         bbox[1]+20 + 15*i),
                                         activefill = "yellow", activewidth = width*2, outline = "green",
                                         width = width, fill = "green", tag = [tag,"nn_draw"]);
                    
                # print('Node___',i);
                
                if l != L-1 and i < nodes_draw[l+1]:
                    
                    weights = self.pacman_brain.getNodeWeights(l, i)[:nodes_draw[l]];
                    
                    # print('ws',len(weights));
                    
                    for w in range(nodes_draw[l]):
                        
                        
                        col = "blue" if weights[w] < 0 else "red";
                        width = max(0.01,abs(weights[w]))*5;
                        tag = f"nn_draw_{l},{w},{i}";
                        
                        if configure_existing:
                            line = self.find_withtag(tag)[0];
                            self.itemconfigure(line, width = width, fill = col);
                        else:
                            bbox = self.bbox_brain;
                            self.create_line(bbox[0]+15 + l*100,
                                             bbox[1]+15 + 15*w,
                                             bbox[0]+15 + (l+1)*100,
                                             bbox[1]+15 + 15*i,
                                             activefill = "yellow", activewidth = width*2,
                                             width = width, fill = col, tag = [tag,"nn_draw"]);
        pass;
    
    def drawGlobals(self):
        '''draws all global stats'''
        
        for stat in self.stats:
            stat_obj = self.find_withtag(stat);
            
            stat_dict = {
                "game_mode_text" : "GAME MODE = {0}".format(self.game_mode),
                "score_text" : "Score: {0}".format(self.score),
                "time_text" : "Time: {0}".format(self.time),
                "speed_text" : "Frame delay: {0}".format(Cons.DELAY),
                "noise_text" : "Noise level: {0}".format(self.noise),
                "items_text" : "Items on canvas: {0}".format(len(self.find_withtag("all"))),
                "key_text" : "Key last pressed: {0}".format(self.key_last_pressed),
                }
            
            self.itemconfigure(stat_obj, text = stat_dict[stat])
        
    def drawBackground(self, image_name):
        '''draws the background of the room'''
        background = self.find_withtag("background");
    
        if len(background) != 0:
            self.itemconfigure(background, image = image_name)
            
        pass;
    
    def gameOver(self):
        '''deletes all objects and draws game over message'''

        self.delete(ALL)
        self.create_text(self.winfo_width() /2, self.winfo_height()/2,
            text="Game Over with score {0}".format(self.score), fill="white")


class Pacman(Frame):

    def __init__(self):
        super().__init__()

        self.master.title('Pacman');
        self.board = Board("pause");
        self.pack();

def main():

    root = Tk()
    root.geometry("1200x800");
    nib = Pacman();
    root.mainloop();
    
if __name__ == '__main__':
    main()