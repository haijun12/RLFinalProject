-- local last_reward_time = 0
-- local reward_cooldown = 1.0
-- local elapsed = 0

if core.settings:has("fixed_map_seed") then
   math.randomseed(core.settings:get("fixed_map_seed"))
end

-- Positive reward if a tree block is dug
-- Track previously dug tree blocks to avoid repeat rewards
local rewarded_tree_positions = {}

minetest.register_on_newplayer(function(player)
    rewarded_tree_positions = {} 
end)

minetest.register_on_dignode(function(pos, node)
  local key = minetest.pos_to_string(pos)
  if string.find(node.name, "tree") and not rewarded_tree_positions[key] then
     set_reward_once(15.0, 0.0)
     rewarded_tree_positions[key] = true
  elseif not string.find(node.name, "tree") then
     set_reward_once(-5.0, 0.0)
  end
end)

-- Positive reward if a tree block is punched
minetest.register_on_punchnode(function(pos, node)
  if string.find(node.name, "tree") then
    set_reward_once(1.0, 0.0)
  else
    set_reward_once(-0.5, 0.0)
  end
end)

local look_start_time = nil
local current_look_pos = nil
local MIN_DIG_TIME = 2.0

minetest.register_globalstep(function(dtime)
    local now = minetest.get_us_time() / 1e6

    for _, player in ipairs(minetest.get_connected_players()) do
        local eye_pos = vector.add(player:get_pos(), {x = 0, y = 1.5, z = 0})
        local dir = player:get_look_dir()
        local look_end = vector.add(eye_pos, vector.multiply(dir, 10))

        for pointed_thing in core.raycast(eye_pos, look_end, false, false) do
            if pointed_thing.type == "node" then
                local node_pos = pointed_thing.under
                local node = minetest.get_node(node_pos)

                if node and string.find(node.name, "tree") then
                    if current_look_pos and vector.equals(current_look_pos, node_pos) then
                        if look_start_time and now - look_start_time > MIN_DIG_TIME then
                            set_reward_once(0.5, 0.0)
                            look_start_time = now + 1000  -- prevent re-reward
                            minetest.log("action", "[SteadyFocus] Sustained look at " ..
                                minetest.pos_to_string(node_pos))
                        end
                    else
                        current_look_pos = node_pos
                        look_start_time = now
                    end
                else
                    current_look_pos = nil
                    look_start_time = nil
                end
                break
            end
        end
    end
end)

-- Track active digging attempt
local current_dig = {
    pos = nil,
    start_time = nil,
    aborted = false
  }
  
  local DIG_TIMEOUT = 2.5  -- seconds allowed to complete a dig
  
  minetest.register_globalstep(function(dtime)
    local now = minetest.get_us_time() / 1e6
  
    for _, player in ipairs(minetest.get_connected_players()) do
      local eye_pos = vector.add(player:get_pos(), {x = 0, y = 1.5, z = 0})
      local dir = player:get_look_dir()
      local look_end = vector.add(eye_pos, vector.multiply(dir, 10))
  
      for pointed in core.raycast(eye_pos, look_end, false, false) do
        if pointed.type == "node" then
          local node_pos = pointed.under
          local node = minetest.get_node(node_pos)
  
          -- If looking at tree and not already digging it
          if node and string.find(node.name, "tree") then
            if current_dig.pos and vector.equals(current_dig.pos, node_pos) then
              -- already tracking, check for timeout
              if current_dig.start_time and (now - current_dig.start_time) > DIG_TIMEOUT then
                set_reward_once(-2.0, 0.0)  -- punish not finishing the dig
                minetest.log("action", "[PartialDig] Timeout digging at " .. minetest.pos_to_string(node_pos))
                current_dig.pos = nil
                current_dig.start_time = nil
              end
            else
              -- start tracking new dig
              current_dig.pos = node_pos
              current_dig.start_time = now
              current_dig.aborted = false
            end
          else
            -- looking away from tracked node before it's dug
            if current_dig.pos and not current_dig.aborted then
              set_reward_once(-1.0, 0.0)
              minetest.log("action", "[PartialDig] Aborted dig at " .. minetest.pos_to_string(current_dig.pos))
              current_dig.aborted = true
            end
          end
          break
        end
      end
    end
  end)
  
  -- Clear tracked dig on completion
  minetest.register_on_dignode(function(pos, node)
    if current_dig.pos and vector.equals(pos, current_dig.pos) then
      current_dig.pos = nil
      current_dig.start_time = nil
      current_dig.aborted = false
    end
  end)
  

local last_proximity_reward_time = 0
local proximity_cooldown = 1.0
local proximity_radius = 10  

minetest.register_globalstep(function(dtime)
    local now = minetest.get_us_time() / 1e6
    if now - last_proximity_reward_time < proximity_cooldown then
        return
    end

    for _, player in ipairs(minetest.get_connected_players()) do
        local pos = player:get_pos()
        local minp = vector.subtract(pos, proximity_radius)
        local maxp = vector.add(pos, proximity_radius)

        local closest_tree_pos = nil
        local closest_distance = math.huge

        for x = math.floor(minp.x), math.floor(maxp.x) do
            for y = math.floor(minp.y), math.floor(maxp.y) do
                for z = math.floor(minp.z), math.floor(maxp.z) do
                    local node_pos = {x = x, y = y, z = z}
                    local node = minetest.get_node_or_nil(node_pos)
                    if node and string.find(node.name, "tree") then
                        local dist = vector.distance(pos, node_pos)
                        if dist < closest_distance then
                            closest_distance = dist
                            closest_tree_pos = node_pos
                        end
                    end
                end
            end
        end

        if closest_tree_pos then
            local max_range = proximity_radius
            local reward = math.max(0.05, 0.2 - (closest_distance / max_range))
            set_reward_once(reward, 0.0)
            last_proximity_reward_time = now
            minetest.log("action", "[Proximity] Closest tree at " .. minetest.pos_to_string(closest_tree_pos) ..
                                      ", reward: " .. reward)
        end
    end
end)

-- Turn on the termination flag if the agent dies
minetest.register_on_dieplayer(function(ObjectRef, reason)
      set_termination()
end)

-- Executed when the player joins the game
minetest.register_on_joinplayer(function(player, _last_login)
      -- set timeofday to midday
      minetest.set_timeofday(0.5)

      -- Disable HUD elements
      player:hud_set_flags({
            hotbar = false,
            crosshair = false,
            healthbar = false,
            chat = false,
      })
end)
