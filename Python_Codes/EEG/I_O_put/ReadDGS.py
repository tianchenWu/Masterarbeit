# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:33:05 2014

@author: wu
"""

import igraph as ig
import copy



class ReadDGS:
    """
    Input: a .dgs file which includes the EEg information of one patient
    Output: ReadDGS object
    Attribute: graphlist
    Usage: ReadDGS(...).graphlist ,so that you can get a list of graph, each of which is a EEG of one patient
    at a specific time stamp
    """
    #type file_content is string
    def __init__(self,file_content):
        self.file_content=file_content
        #a list of igraph objects in time series
        self.graphlist=[]
        self.load()
        #1st graph is only for setting nodes
        self.graphlist=self.graphlist[1:]
    
    #read in the whole dgs file
    def load(self):
        for line in self.file_content.split("\n"):
            self.readbyline(line)
            
    
    #read in one line of dgs file
    def readbyline(self,line):
        tokens=line.split()
        #check empty list
        if not tokens:
            return
        
        # beginning of a time stamp
        if tokens[0]=="st":
            #create a new graph object if the graphlist is empty
            if not self.graphlist:
                graph=ig.Graph()
            else:
            #otherwise make a copy of the origianl graph
                graph=copy.deepcopy(self.graphlist[-1])
            
            #add the sequential number of st as a attribute of graph
            graph["st"]=tokens[1]
            #add the new created graph to the end of the graphlist
            self.graphlist.append(graph)
            
        #add node
        elif tokens[0]=="an":
            #process list tokens[2:] so that it can be accepted by add_vertex(name,**kwds)
            dict={}
            for token in tokens[2:]:
                key,value=token.split("=")
                #print "key: %s, value: %s" %(key,value)
                dict[key]=value
            #add vertex
            #print dict
            self.graphlist[-1].add_vertex(name=tokens[1],**dict)
            #print self.graphlist[-1].vs[0].attributes()
            
        
        #add edge
        elif tokens[0]=="ae":
            op,edge_name,source,target,weight_str=tuple(tokens)
            dict={}
            dict["edge_name"]=edge_name
            dict["weight"]=float(weight_str.split("=")[1])
            #add edge
            self.graphlist[-1].add_edge(source,target,**dict)
        
        #change edge
        elif tokens[0]=="ce":
            op,edge_name,weight_str=tuple(tokens)
            weight,value=weight_str.split("=")
            self.graphlist[-1].es.select(edge_name=edge_name)[weight]=float(value)
        
        #delete edge
        elif tokens[0]=="de":
            op,edge_name=tuple(tokens)
            self.graphlist[-1].es.select(edge_name=edge_name).delete()
        
        #change node
        elif tokens[0]=="cn":
            ("there is a cn")
            
        else:
            pass
    
    #visualize the graph
    """
    def draw(self):
        for graph in self.graphlist:
            layout=graph.layout("kk")
            ig.plot(graph,layout=layout)
    """
    
    
 